"""What a baseline ran: source, dependencies, toolchain and the native libraries it built.

    python scripts/baseline_identity.py OUT.json

Writes one JSON record. ``kit_environment`` is the GPU kit's own record of this
process (``scripts/gpu/gpukit.py``: versions, devices, the probe, allowlisted and
redacted environment variables); the rest is what the kit doesn't record and a CPU
baseline depends on: the git trees of the library and both Rust crates, the Python
packages, the Rust and C toolchains, and the SHA-256 of each native library that
``scripts/baseline.sh`` builds (none of which is built here). Version skew between
these is what an identical commit can still hide, so every field is read, not assumed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "gpu"))

import gpukit  # noqa: E402 - after the path change

from gpubridge import multitcp, verbs  # noqa: E402

PACKAGES = ("torch", "numpy", "pytest", "pytest-timeout", "triton", "gpubridge")
TREES = ("src", "src/gpubridge/verbs/crate", "src/gpubridge/multitcp/crate", "scripts", "tests")


def _command(*args: str) -> str | None:
    """A command's stdout, stripped, or None if it is missing or fails."""
    if shutil.which(args[0]) is None:
        return None
    result = subprocess.run(args, capture_output=True, text=True, cwd=REPO, timeout=60)
    return result.stdout.strip() if result.returncode == 0 else None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _library(path: Path) -> dict | None:
    """The built library at ``path``: its name, size and SHA-256, or None if absent."""
    if not path.exists():
        return None
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def identity() -> dict:
    libraries = {"multitcp": _library(multitcp.library_path()),
                 "verbs_test_fabric": _library(verbs.library_path(test_fabric=True))}
    if sys.platform.startswith("linux"):
        libraries["verbs"] = _library(verbs.library_path())
    status = _command("git", "status", "--porcelain", "--untracked-files=no")
    cc = _command("cc", "--version")
    return {
        "git": {
            "commit": _command("git", "rev-parse", "HEAD"),
            "branch": _command("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
            # The committed trees; with dirty true, the working tree differs from them.
            "trees": {tree: _command("git", "rev-parse", f"HEAD:{tree}") for tree in TREES},
        },
        "python": {"version": platform.python_version(), "implementation":
                   platform.python_implementation(), "executable": Path(sys.executable).name},
        "packages": {name: _version(name) for name in PACKAGES},
        "toolchain": {
            "rustup_toolchain": os.environ.get("RUSTUP_TOOLCHAIN"),
            "rustc": _command("rustc", "-Vv"),
            "cargo": _command("cargo", "-V"),
            "cc": cc.splitlines()[0] if cc else None,
        },
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "mac_ver": platform.mac_ver()[0] or None,
                     "libc": " ".join(platform.libc_ver()).strip() or None},
        "libraries": libraries,
        "kit_environment": gpukit.environment(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    gpukit.write_json(out, identity())
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
