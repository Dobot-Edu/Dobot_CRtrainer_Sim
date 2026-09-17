import enum
import copy
import h5py
import os
import re

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData


class StreamWriteMode(enum.Enum):
    APPEND = 0  # Append the record
    LAST = 1    # Write the last record


class StreamingHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    def __init__(self):
        """
            compression options:
            - gzip: high compression ratio (50-80%), high latency due to CPU-intensive compression
            - lzf: moderate compression ratio (30-50%), low latency, fast compression algorithm
            - None: don't use compression, will cause minimum latency but largest file size
        """
        super().__init__()
        self._chunks_length = 100
        self._compression = None
        self._writer = self.SingleThreadHDF5DatasetWriter(self)

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        if resume:
            self._hdf5_file_stream = h5py.File(file_path, "a")
            self._hdf5_data_group = self._hdf5_file_stream["data"]
            self._demo_count = len(self._hdf5_data_group)
        else:
            self._hdf5_file_stream = h5py.File(file_path, "w")
            # set up a data group in the file
            self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
            self._hdf5_data_group.attrs["total"] = 0
            self._demo_count = 0

            env_name = env_name if env_name is not None else ""
            self.add_env_args({"env_name": env_name, "type": 2})

    class SingleThreadHDF5DatasetWriter:
        def __init__(self, file_handler):
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.file_handler = file_handler

        def write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData, write_mode: StreamWriteMode):
            episode_copy = copy.deepcopy(episode)
            funture = self.executor.submit(self._do_write_episode, h5_episode_group, episode_copy)
            return funture.result() if write_mode == StreamWriteMode.LAST else funture

        def _do_write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData):
            def create_dataset_helper(group, key, value):
                """Helper method to create dataset that contains recursive dict objects."""
                if isinstance(value, dict):
                    if key not in group:
                        key_group = group.create_group(key)
                    else:
                        key_group = group[key]
                    for sub_key, sub_value in value.items():
                        create_dataset_helper(key_group, sub_key, sub_value)
                else:
                    data = value.cpu().numpy()
                    if key not in group:
                        dataset = group.create_dataset(
                            key,
                            shape=data.shape,
                            maxshape=(None, *data.shape[1:]),
                            chunks=(self.file_handler.chunks_length, *data.shape[1:]),
                            dtype=data.dtype,
                            compression=self.file_handler.compression,
                        )
                        dataset[0: data.shape[0]] = data
                    else:
                        dataset = group[key]
                        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
                        dataset[dataset.shape[0] - data.shape[0]:] = data

            for key, value in episode.data.items():
                create_dataset_helper(h5_episode_group, key, value)

            self.file_handler.flush()

        def shutdown(self):
            self.executor.shutdown(wait=True)

    @property
    def chunks_length(self) -> int:
        return self._chunks_length

    @chunks_length.setter
    def chunks_length(self, chunks_length: int):
        self._chunks_length = chunks_length

    @property
    def compression(self) -> str | None:
        return self._compression

    @compression.setter
    def compression(self, compression: str | None):
        self._compression = compression

    def write_episode(self, episode: EpisodeData, write_mode: StreamWriteMode) -> bool:
        self._raise_if_not_initialized()

        group_name = f"demo_{self._demo_count}"
        group_exists = group_name in self._hdf5_data_group
        # A 100-frame streaming flush can leave no buffered tensors at the
        # exact R/N boundary.  LAST must still finalize the already-written
        # group so its success/total metadata is not lost.
        if episode.is_empty() and not (write_mode == StreamWriteMode.LAST and group_exists):
            return False

        if group_name not in self._hdf5_data_group:
            h5_episode_group = self._hdf5_data_group.create_group(group_name)
        else:
            h5_episode_group = self._hdf5_data_group[group_name]

        # store number of steps taken
        if "actions" in episode.data:
            if "num_samples" not in h5_episode_group.attrs:
                h5_episode_group.attrs["num_samples"] = 0
            h5_episode_group.attrs["num_samples"] += len(episode.data["actions"])
        elif "num_samples" not in h5_episode_group.attrs:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        if write_mode == StreamWriteMode.LAST:
            # increment total step counts
            self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

            # increment total demo counts
            self._demo_count += 1

        self._writer.write_episode(h5_episode_group, episode, write_mode)
        return True

    def close(self):
        self._writer.shutdown()
        super().close()


class PerEpisodeStreamingHDF5DatasetFileHandler(StreamingHDF5DatasetFileHandler):
    """Write one complete episode per task-name-plus-index HDF5 file."""

    _INDEX_WIDTH = 6

    def __init__(self):
        super().__init__()
        self._file_prefix: Path | None = None
        self._next_episode_index = 0
        self._existing_episode_count = 0
        self._completed_episode_count = 0
        self._persistent_env_args: dict = {}
        self._current_file_path: Path | None = None

    def create(self, file_path: str, env_name: str = None, resume: bool = False):
        """Configure lazy episode files and continue after existing indices.

        ``resume`` is accepted for interface compatibility.  Per-episode files
        are immutable after an R/N boundary, so a new index is always used.
        """
        if self._file_prefix is not None:
            raise RuntimeError("Per-episode HDF5 file handler is already configured")

        prefix = Path(file_path)
        if prefix.suffix.lower() == ".hdf5":
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._file_prefix = prefix
        if env_name:
            self._persistent_env_args.update({"env_name": env_name, "type": 2})

        pattern = re.compile(
            rf"^{re.escape(prefix.name)}_(\d{{{self._INDEX_WIDTH}}})\.hdf5$"
        )
        existing_indices = []
        for candidate in prefix.parent.glob(f"{prefix.name}_*.hdf5"):
            match = pattern.match(candidate.name)
            if match:
                existing_indices.append(int(match.group(1)))
        self._existing_episode_count = len(existing_indices)
        self._next_episode_index = max(existing_indices, default=-1) + 1

    def _open_next_episode_file(self) -> None:
        if self._file_prefix is None:
            raise RuntimeError("Per-episode HDF5 file handler is not configured")
        if self._hdf5_file_stream is not None:
            return

        current_path = self._file_prefix.with_name(
            f"{self._file_prefix.name}_{self._next_episode_index:0{self._INDEX_WIDTH}d}.hdf5"
        )
        # The startup scan should already make this unique.  Keep the final
        # guard to avoid overwriting a file created by another process.
        while current_path.exists():
            self._next_episode_index += 1
            current_path = self._file_prefix.with_name(
                f"{self._file_prefix.name}_{self._next_episode_index:0{self._INDEX_WIDTH}d}.hdf5"
            )

        self._current_file_path = current_path
        super().create(
            str(current_path),
            env_name=self._persistent_env_args.get("env_name"),
            resume=False,
        )
        if self._persistent_env_args:
            super().add_env_args(dict(self._persistent_env_args))
        print(f"[CR5_Sim] Recording episode HDF5: {current_path}")

    def _close_current_stream(self) -> None:
        if self._hdf5_file_stream is None:
            return
        self._hdf5_file_stream.flush()
        self._hdf5_file_stream.close()
        self._hdf5_file_stream = None
        self._hdf5_data_group = None

    def add_env_args(self, env_args: dict):
        self._persistent_env_args.update(env_args)
        if self._hdf5_file_stream is not None:
            super().add_env_args(env_args)

    def write_episode(self, episode: EpisodeData, write_mode: StreamWriteMode) -> bool:
        self._open_next_episode_file()
        completed = super().write_episode(episode, write_mode)
        if write_mode == StreamWriteMode.LAST and completed:
            completed_path = self._current_file_path
            self._close_current_stream()
            self._completed_episode_count += 1
            self._next_episode_index += 1
            self._current_file_path = None
            print(f"[CR5_Sim] Saved episode HDF5: {completed_path}")
        return completed

    def get_num_episodes(self) -> int:
        return self._existing_episode_count + self._completed_episode_count

    @property
    def next_episode_index(self) -> int:
        return self._next_episode_index

    def close(self):
        # If the process stops mid-episode, close a valid partial HDF5.  It has
        # no success attribute and can be skipped by conversion/validation.
        self._writer.shutdown()
        self._close_current_stream()
