"""LeRobot async-inference client specialized for CR5A plus AG95."""

from __future__ import annotations

import pickle
import time
from collections.abc import Mapping

import grpc
import numpy as np
import torch

from .lerobot_compat import (
    RemotePolicyConfig,
    TimedObservation,
    install_pickle_compatibility,
)
from .transport import services_pb2, services_pb2_grpc
from .transport.utils import grpc_channel_options, send_bytes_in_chunks


CR5A_AXIS_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_finger1_joint",
)


def _rgb_uint8(value: torch.Tensor, key: str) -> np.ndarray:
    """Convert one batched Isaac RGB observation to contiguous HWC uint8."""

    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Camera {key!r} must have shape (1,H,W,C) or (H,W,C); got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Camera {key!r} must have three RGB channels; got {array.shape}")

    if array.dtype != np.uint8:
        array = np.asarray(array, dtype=np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


class CR5ALeRobotPolicyClient:
    """Synchronous CR5A client for LeRobot's asynchronous policy server.

    Each request contains a 7-D absolute joint state and the wrist/front RGB
    images.  Returned actions remain 7-D absolute joint targets in radians; no
    SO101 or X-Trainer unit conversion is applied.
    """

    def __init__(
        self,
        host: str,
        port: int,
        camera_infos: Mapping[str, tuple[int, int]],
        pretrained_name_or_path: str,
        *,
        policy_type: str = "act",
        actions_per_chunk: int = 10,
        device: str = "cuda",
        rpc_timeout_s: float = 5.0,
        setup_timeout_s: float = 180.0,
        task_description: str | None = None,
    ) -> None:
        if actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be positive")
        if set(camera_infos) != {"wrist", "front"}:
            raise ValueError(
                f"CR5A ACT expects exactly wrist/front cameras; got {sorted(camera_infos)}"
            )

        install_pickle_compatibility()

        self.actions_per_chunk = actions_per_chunk
        self.rpc_timeout_s = rpc_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.task_description = task_description
        self._timestep = 0
        self._last_hold = np.zeros(len(CR5A_AXIS_NAMES), dtype=np.float32)

        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(CR5A_AXIS_NAMES),),
                "names": list(CR5A_AXIS_NAMES),
            }
        }
        for key in ("wrist", "front"):
            height, width = camera_infos[key]
            features[f"observation.images.{key}"] = {
                "dtype": "image",
                "shape": (int(height), int(width), 3),
                "names": ["height", "width", "channels"],
            }

        self.policy_config = RemotePolicyConfig(
            policy_type=policy_type,
            pretrained_name_or_path=pretrained_name_or_path,
            lerobot_features=features,
            actions_per_chunk=actions_per_chunk,
            device=device,
        )
        self.channel = grpc.insecure_channel(
            f"{host}:{port}",
            grpc_channel_options(),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self._initialize_server()

    def _initialize_server(self) -> None:
        try:
            self.stub.Ready(services_pb2.Empty(), timeout=self.rpc_timeout_s)
            payload = pickle.dumps(self.policy_config)
            print("[Policy] Loading checkpoint on the LeRobot server...")
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=payload),
                timeout=self.setup_timeout_s,
            )
        except grpc.RpcError as exc:
            self.channel.close()
            raise RuntimeError(
                f"Could not initialize the LeRobot policy server: {exc.code().name}: {exc.details()}"
            ) from exc
        print("[Policy] LeRobot server is ready.")

    @staticmethod
    def _joint_state(observation_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        if "joint_pos" not in observation_dict:
            raise KeyError("Policy observation is missing 'joint_pos'")
        joint_pos = observation_dict["joint_pos"].detach().cpu().numpy().astype(np.float32)
        joint_pos = joint_pos.reshape(-1)
        if joint_pos.shape != (len(CR5A_AXIS_NAMES),):
            raise ValueError(f"Expected 7-D joint_pos, got {joint_pos.shape}")
        if not np.isfinite(joint_pos).all():
            raise ValueError("joint_pos contains NaN or Inf")
        return joint_pos

    def reset(self, joint_pos: torch.Tensor | np.ndarray | None = None) -> None:
        """Clear local fallback state after an environment reset."""

        if joint_pos is None:
            self._last_hold.fill(0.0)
            return
        value = np.asarray(
            joint_pos.detach().cpu().numpy() if isinstance(joint_pos, torch.Tensor) else joint_pos,
            dtype=np.float32,
        ).reshape(-1)
        if value.shape != (len(CR5A_AXIS_NAMES),) or not np.isfinite(value).all():
            raise ValueError("reset joint_pos must be a finite 7-D vector")
        self._last_hold = value.copy()

    def _raw_observation(self, observation_dict: Mapping[str, torch.Tensor]) -> dict:
        joint_pos = self._joint_state(observation_dict)
        self._last_hold = joint_pos.copy()

        raw: dict = {
            "wrist": _rgb_uint8(observation_dict["wrist"], "wrist"),
            "front": _rgb_uint8(observation_dict["front"], "front"),
        }
        raw.update(zip(CR5A_AXIS_NAMES, joint_pos.tolist(), strict=True))
        if self.task_description:
            raw["task"] = self.task_description
        return raw

    def _send_observation(self, observation_dict: Mapping[str, torch.Tensor]) -> None:
        self._timestep += 1
        # This client replans synchronously every action chunk.  must_go avoids
        # LeRobot filtering a new camera frame merely because joint motion was
        # smaller than its generic observation-distance threshold.
        observation = TimedObservation(
            timestamp=time.time(),
            timestep=self._timestep,
            observation=self._raw_observation(observation_dict),
            must_go=True,
        )
        iterator = send_bytes_in_chunks(
            pickle.dumps(observation),
            services_pb2.Observation,
            log_prefix="[CR5A] Observation",
            silent=True,
        )
        self.stub.SendObservations(iterator, timeout=self.rpc_timeout_s)

    def _hold_chunk(self) -> torch.Tensor:
        hold = torch.from_numpy(self._last_hold.copy())
        return hold.repeat(self.actions_per_chunk, 1).unsqueeze(1)

    def get_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Request one action chunk, falling back to a current-position hold."""

        try:
            self._send_observation(observation_dict)
            reply = self.stub.GetActions(services_pb2.Empty(), timeout=self.rpc_timeout_s)
        except grpc.RpcError as exc:
            print(f"[Policy] RPC {exc.code().name}; holding current joint positions.")
            return self._hold_chunk()

        if not reply.data:
            print("[Policy] Server returned no action; holding current joint positions.")
            return self._hold_chunk()

        try:
            timed_actions = pickle.loads(reply.data)  # trusted local policy server
            vectors = []
            for timed_action in timed_actions:
                vector = torch.as_tensor(timed_action.get_action(), dtype=torch.float32).cpu().reshape(-1)
                if vector.shape != (len(CR5A_AXIS_NAMES),):
                    raise ValueError(f"expected 7-D action, got {tuple(vector.shape)}")
                vectors.append(vector)
            if not vectors:
                raise ValueError("empty action list")
            chunk = torch.stack(vectors, dim=0)
        except Exception as exc:
            print(f"[Policy] Invalid action payload ({exc}); holding current joint positions.")
            return self._hold_chunk()

        return chunk.unsqueeze(1)

    def close(self) -> None:
        self.channel.close()


__all__ = ["CR5ALeRobotPolicyClient", "CR5A_AXIS_NAMES"]
