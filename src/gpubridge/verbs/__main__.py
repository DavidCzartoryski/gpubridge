"""``python -m gpubridge.verbs build`` compiles the verbs bridge; ``info`` says what it is.

Build once where a Rust toolchain and rdma-core's headers are (``cargo`` on PATH;
``GPUBRIDGE_VERBS_INCLUDE_DIR`` if the headers aren't installed), so jobs don't
compile it on first use, or build it on a machine of the same architecture and
set ``GPUBRIDGE_VERBS_LIB`` to the library; ``build`` then only checks that it
loads. ``info`` loads the library and opens no device.
"""

from __future__ import annotations

import argparse
import os
import sys

from gpubridge import verbs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gpubridge.verbs", description=__doc__)
    parser.add_argument("command", choices=["build", "info"])
    parser.add_argument("--force", action="store_true", help="rebuild even if built")
    args = parser.parse_args(argv)
    try:
        if args.command == "build" and not os.environ.get(verbs.LIB_ENV, "").strip():
            print(verbs.build(force=args.force))
            return 0
        lib = verbs.load(test_fabric=False)
        print(f"library: {lib._name}\n{verbs.build_info(lib)}\nprotocol {verbs.protocol(lib)}")
        return 0
    except verbs.Unavailable as exc:
        print(f"verbs bridge unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
