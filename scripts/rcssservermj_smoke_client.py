from __future__ import annotations

import argparse
import socket
import sys


def send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(len(payload).to_bytes(4, byteorder="big", signed=False) + payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_framed(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    size = int.from_bytes(header, byteorder="big", signed=False)
    if size <= 0 or size > 10_000_000:
        raise ValueError(f"invalid RCSSServerMJ frame size: {size}")
    return recv_exact(sock, size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    init = b"(init T1 RoboCOP 1)"
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_framed(sock, init)
        reply = receive_framed(sock)

    text = reply.decode("utf-8", errors="replace")
    if not text.strip():
        print("FAIL: RCSSServerMJ returned an empty perception frame", file=sys.stderr)
        return 2

    print("PASS: RCSSServerMJ accepted a T1 agent and returned a perception frame")
    print(f"frame_bytes={len(reply)}")
    print(f"frame_prefix={text[:180]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
