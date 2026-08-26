"""Resolve canonical runtime paths in the monorepo and root-hoisted public mirror."""

from __future__ import annotations

import sys
from pathlib import Path

_HOISTED = {
    "adapters/codex/app_server.py": "app_server.py",
    "adapters/codex/test_app_server.py": "test_app_server.py",
    "adapters/llm/providers.py": "providers.py",
    "adapters/llm/test_providers.py": "test_providers.py",
    "adapters/voice_pm/publisher.py": "publisher.py",
    "adapters/voice_pm/test_publisher.py": "test_publisher.py",
    "adapters/voice_pm/test_switch.py": "test_switch.py",
    "adapters/voice_pm/test_wiring.py": "test_wiring.py",
    "adapters/voice_pm/wiring.py": "wiring.py",
    "contracts/events.py": "events.py",
}


def repository_root(anchor: Path) -> Path:
    """Return the source root for either ``voice/file.py`` or a hoisted file."""
    resolved = anchor.resolve()
    return resolved.parent.parent if resolved.parent.name == "voice" else resolved.parent


def logical_path(root: Path, relative: str) -> Path:
    """Map a release-manifest path to its source without changing bundle paths."""
    canonical = root / relative
    if canonical.exists():
        return canonical
    if relative.startswith("voice/"):
        hoisted = root / relative.removeprefix("voice/")
        if hoisted.exists():
            return hoisted
    alias = _HOISTED.get(relative)
    if alias is not None and (root / alias).exists():
        return root / alias
    return canonical


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: repository_layout.py LOGICAL_PATH")
    print(logical_path(repository_root(Path(__file__)), sys.argv[1]))
