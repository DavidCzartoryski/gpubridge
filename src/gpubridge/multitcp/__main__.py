"""``python -m gpubridge.multitcp build`` compiles the multitcp bridge; ``info`` says what it is.

On a cluster, run ``build`` once where a Rust toolchain is (``cargo`` on PATH),
e.g. in the kit's setup job, so jobs don't compile it on first use. Without one
on the nodes, build it on a machine of the same architecture and set
``GPUBRIDGE_MULTITCP_LIB`` to the library.
"""

from __future__ import annotations

import argparse
import sys

from gpubridge import multitcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gpubridge.multitcp", description=__doc__)
    parser.add_argument("command", choices=["build", "info"])
    parser.add_argument("--force", action="store_true", help="rebuild even if built")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            print(multitcp.build(force=args.force))
            return 0
        print(f"library: {multitcp.load()._name}\nprotocol {multitcp.protocol()}")
        return 0
    except multitcp.Unavailable as exc:
        print(f"multitcp bridge unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
