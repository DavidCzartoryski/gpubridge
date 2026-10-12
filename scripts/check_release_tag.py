"""Check that a release tag names the package's version.

    python scripts/check_release_tag.py v0.3.0-alpha

Tags read vX.Y.Z, optionally with -alpha, -beta or -rc and a number
(v0.3.0-alpha is 0.3.0a0, v1.0.0-rc2 is 1.0.0rc2). The version is the one in
pyproject.toml. Exits non-zero, saying both, when they differ, so the release
workflow never builds a package whose version doesn't match its tag.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PRE = {"alpha": "a", "beta": "b", "rc": "rc"}


def version_of_tag(tag: str) -> str:
    """The PEP 440 version a release tag stands for."""
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)(\d*))?", tag)
    if match is None:
        raise ValueError(f"{tag!r} is not a release tag like v0.3.0 or v0.3.0-alpha")
    release, pre, number = match.groups()
    return release + (f"{PRE[pre]}{number or 0}" if pre else "")


def project_version(pyproject: Path = PYPROJECT) -> str:
    match = re.search(r'^version = "([^"]+)"$', pyproject.read_text(), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"no version in {pyproject}")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        wanted, version = version_of_tag(args[0]), project_version()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if wanted != version:
        print(f"tag {args[0]} means version {wanted}, but pyproject.toml says {version}",
              file=sys.stderr)
        return 1
    print(f"tag {args[0]} matches version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
