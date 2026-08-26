#!/usr/bin/env python3
"""Exercise CI collection and representative runtime tests from a clean clone."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="zer0-voice-clean-clone-") as directory:
        clone = Path(directory) / "checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--local",
                "--no-hardlinks",
                str(source),
                str(clone),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "test_repository_layout",
                "test_release",
                "test_simple",
                "test_workspace_router",
                "test_providers",
            ],
            cwd=clone,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
