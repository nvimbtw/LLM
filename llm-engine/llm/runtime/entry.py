"""Fresh-interpreter launch shim; executable startup is gated by its recorded owner."""

import os
from pathlib import Path
import sys


def wait_for_start(descriptor: int) -> None:
    """Refuse custom execution if its creator died before fsyncing ownership and releasing startup."""
    try:
        if os.read(descriptor, 1) != b"1":
            raise SystemExit(1)
    finally:
        os.close(descriptor)


def main() -> None:
    """Select a fixed trusted role without importing custom backends in the parent or guardian."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    role, directory, *values = sys.argv[1:]
    if role == "guardian":
        data, control, lock, start = map(int, values)
        wait_for_start(start)
        from llm.runtime.guardian import run_guardian

        run_guardian(Path(directory), data, control, lock)
    elif role == "worker":
        data, start, guardian = map(int, values)
        from llm.runtime.processes import require_guardian

        require_guardian(guardian)
        wait_for_start(start)
        from llm.runtime.worker import run_worker

        run_worker(Path(directory), data)
    else:
        raise SystemExit("unknown runtime role")


if __name__ == "__main__":
    main()
