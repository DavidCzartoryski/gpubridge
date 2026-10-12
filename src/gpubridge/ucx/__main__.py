"""``python -m gpubridge.ucx build`` compiles the UCX bridge; ``info`` says what it can do.

On a cluster, run ``build`` once in the job's environment (after ``module load
ucx`` or with ``UCX_HOME`` set), e.g. in the kit's setup job, so jobs don't
compile it on first use.
"""

from __future__ import annotations

import argparse
import sys

from gpubridge import ucx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gpubridge.ucx", description=__doc__)
    parser.add_argument("command", choices=["build", "info"])
    parser.add_argument("--force", action="store_true", help="rebuild even if built")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            print(ucx.build(force=args.force))
            return 0
        worker = ucx.Worker(1)
        try:
            types = ", ".join(sorted(worker.memory_types()))
        finally:
            worker.close()
        major, minor, release = ucx.version()
        print(f"library: {ucx.library_path()}\nUCX {major}.{minor}.{release}\n"
              f"memory types: {types}")
        return 0
    except ucx.Unavailable as exc:
        print(f"UCX bridge unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
