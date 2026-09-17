"""Minimal gRPC transport shared with LeRobot's async inference server."""

from . import services_pb2, services_pb2_grpc

__all__ = ["services_pb2", "services_pb2_grpc"]
