from __future__ import annotations

import argparse
import socket
import sys


def send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(len(payload).to_bytes(4, byteorder="big", signed=False) + payload)


def receive_framed(sock: socket.socket) -> bytes:
    header = bytearray(4)
    if sock.recv_into(header, nbytes=4, flags=socket.MSG_WAITALL) != 4:
        raise ConnectionResetError("server closed before frame header")
    size = int.from_bytes(header, byteorder="big", signed=False)
    if size <= 0 or size > 10_000_000:
        raise ValueError(f"invalid RCSSServerMJ frame size: {size}")
    payload = bytearray(size)
    if sock.recv_into(payload, nbytes=size, flags=socket.MSG_WAITALL) != size:
        raise ConnectionResetError("server closed before complete perception frame")
    return bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=60000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--model", default="T1")
    parser.add_argument("--team", default="RoboCOP")
    parser.add_argument("--player", default="1")
    args = parser.parse_args()

    # Keep this byte-for-byte equivalent to the upstream RCSSServerMJ minimal
    # client protocol: (init <model> <team> <player_no>), 4-byte BE framing.
    init = f"(init {args.model} {args.team} {args.player})".encode()
    print(f"INIT: {init.decode()}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(args.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((args.host, args.port))
            send_framed(sock, init)
            reply = receive_framed(sock)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    text = reply.decode("utf-8", errors="replace")
    if not text.strip():
        print("FAIL: RCSSServerMJ returned an empty perception frame", file=sys.stderr)
        return 3

    print("PASS: RCSSServerMJ accepted the agent and returned a perception frame")
    print(f"frame_bytes={len(reply)}")
    print(f"frame_prefix={text[:300]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
