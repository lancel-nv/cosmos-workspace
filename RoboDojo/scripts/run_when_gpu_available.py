from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
import subprocess
import time


def gpu_status(index):
    output = subprocess.check_output(
        ["nvidia-smi", "--id", str(index), "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    fields = output.strip().split(",")
    if len(fields) != 2:
        raise ValueError(f"Unexpected GPU status: {output!r}")
    return int(fields[0]), int(fields[1])


def main():
    parser = argparse.ArgumentParser(
        description="Wait for GPU memory without stopping other workloads, then exec a command"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=32768)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.gpu < 0 or args.min_free_mib <= 0 or args.interval <= 0 or args.consecutive <= 0:
        parser.error("Provide a command and positive wait settings")
    ready = 0
    while ready < args.consecutive:
        try:
            free, utilization = gpu_status(args.gpu)
            ready = ready + 1 if free >= args.min_free_mib else 0
            print(
                f"{datetime.now(UTC).isoformat()} GPU_WAIT gpu={args.gpu} free_mib={free} "
                f"utilization={utilization} ready={ready}/{args.consecutive}",
                flush=True,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            ready = 0
            print(f"GPU_WAIT_QUERY_ERROR {error}", flush=True)
        if ready < args.consecutive:
            time.sleep(args.interval)
    print(f"GPU_READY exec={command!r}; no other workloads stopped", flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
