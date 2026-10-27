"""Bounded Unix packet transport for the versioned worker protocol."""

import socket

from llm.runtime.protocol import MAX_RUNTIME_PACKET_BYTES, RuntimeEnvelope


def socket_pair(pending: int = 64) -> tuple[socket.socket, socket.socket]:
    """Create bounded independent packet endpoints; callers own closing both handles."""
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        for channel in (first, second):
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, pending * MAX_RUNTIME_PACKET_BYTES)
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, pending * MAX_RUNTIME_PACKET_BYTES)
            channel.settimeout(1.0)
    except BaseException:
        first.close()
        second.close()
        raise
    return first, second


def receive(channel: socket.socket) -> RuntimeEnvelope:
    """Read one packet, rejecting kernel truncation, peer closure and invalid protocol bytes."""
    packet, _, flags, _ = channel.recvmsg(MAX_RUNTIME_PACKET_BYTES)
    if not packet:
        raise EOFError("runtime peer closed")
    if flags & socket.MSG_TRUNC:
        raise ValueError("runtime_packet_truncated")
    return RuntimeEnvelope.from_packet(packet)


def send(channel: socket.socket, envelope: RuntimeEnvelope) -> None:
    """Send one validated packet under the endpoint's finite write timeout."""
    channel.sendall(envelope.to_packet())
