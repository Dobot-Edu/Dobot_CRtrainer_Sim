"""Byte chunking and channel settings used by the async inference protocol."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import Any

from . import services_pb2


CHUNK_SIZE = 2 * 1024 * 1024
MAX_MESSAGE_SIZE = 4 * 1024 * 1024


def send_bytes_in_chunks(
    payload: bytes,
    message_class: Any,
    log_prefix: str = "",
    silent: bool = True,
) -> Iterator[Any]:
    del log_prefix, silent
    buffer = io.BytesIO(payload)
    size = len(payload)
    sent = 0
    while sent < size:
        if sent + CHUNK_SIZE >= size:
            state = services_pb2.TransferState.TRANSFER_END
        elif sent == 0:
            state = services_pb2.TransferState.TRANSFER_BEGIN
        else:
            state = services_pb2.TransferState.TRANSFER_MIDDLE
        chunk = buffer.read(min(CHUNK_SIZE, size - sent))
        yield message_class(transfer_state=state, data=chunk)
        sent += len(chunk)


def grpc_channel_options() -> list[tuple[str, int | str]]:
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
    return [
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", json.dumps(service_config)),
    ]


__all__ = ["grpc_channel_options", "send_bytes_in_chunks"]
