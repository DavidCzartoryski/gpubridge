"""Stop a job whose ``import gpubridge`` isn't this checkout's src/gpubridge.

The kit runs its scripts from one checkout but imports gpubridge from wherever the
venv (an editable install of some checkout) or PYTHONPATH points. When those
differ, a run measures code other than the commit its records name. common.sh's
require_kit_import runs this after activating a venv, and probe.py runs the same
check on every side of a job; both stop the job on a mismatch.

    python scripts/gpu/import_check.py     # exit 0 and one line, or exit 1 and why
"""

from __future__ import annotations

import os
import sys

from gpukit import kit_import, redact

MARKER = "gpubridge-kit: import check failed"


def message(info: dict) -> str | None:
    """None if the import is this checkout's; else what's wrong and how to fix it."""
    if info["gpubridge_in_kit_checkout"]:
        return None
    return (f"{MARKER}: gpubridge was imported from {info['gpubridge_file']} "
            f"(commit {info['gpubridge_git_commit'] or 'unknown'}), not from the checkout "
            f"whose scripts are running ({info['kit_checkout_src']}). PYTHONPATH is "
            f"{os.environ.get('PYTHONPATH') or 'unset'}. Run the job from the checkout the "
            "venv installs, or put this checkout's src first on PYTHONPATH.")


def main() -> int:
    info = kit_import()
    problem = message(info)
    if problem:
        print(redact(problem), file=sys.stderr)
        return 1
    print(redact(f"gpubridge from {info['gpubridge_file']} "
                 f"(commit {info['gpubridge_git_commit']}, dirty {info['gpubridge_git_dirty']})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
