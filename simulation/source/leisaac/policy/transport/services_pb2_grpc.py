"""Client stub for the LeRobot AsyncInference gRPC service."""

import grpc

from . import services_pb2


class AsyncInferenceStub:
    def __init__(self, channel: grpc.Channel) -> None:
        self.SendObservations = channel.stream_unary(
            "/transport.AsyncInference/SendObservations",
            request_serializer=services_pb2.Observation.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
            _registered_method=True,
        )
        self.GetActions = channel.unary_unary(
            "/transport.AsyncInference/GetActions",
            request_serializer=services_pb2.Empty.SerializeToString,
            response_deserializer=services_pb2.Actions.FromString,
            _registered_method=True,
        )
        self.SendPolicyInstructions = channel.unary_unary(
            "/transport.AsyncInference/SendPolicyInstructions",
            request_serializer=services_pb2.PolicySetup.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
            _registered_method=True,
        )
        self.Ready = channel.unary_unary(
            "/transport.AsyncInference/Ready",
            request_serializer=services_pb2.Empty.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
            _registered_method=True,
        )


__all__ = ["AsyncInferenceStub"]
