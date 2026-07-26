#!/usr/bin/env python3
"""Immutable, content-addressed release boundary for the voice runtime."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
MINIMUM_TURNS = 10
PRODUCTION_PIPELINE = "codex-continuous-pcm-v5"
MANIFEST = "manifest.json"
RUNTIME_FILES = (
    "voice/candidate-service",
    "voice/duplex",
    "voice/duplex.py",
    "voice/test_duplex.py",
    "voice/conversation.py",
    "voice/control_plane.py",
    "voice/floor.py",
    "voice/health.py",
    "voice/handoff.py",
    "voice/test_handoff.py",
    "voice/test_health.py",
    "voice/indicator.py",
    "voice/lexicon.json",
    "voice/lexicon.py",
    "voice/modes.py",
    "voice/preflight.py",
    "voice/release.py",
    "voice/runtime_manager.py",
    "voice/test_runtime_manager.py",
    "voice/routes.json",
    "voice/shadow.py",
    "voice/shadow_real.py",
    "voice/shadow-real",
    "voice/test_shadow.py",
    "voice/test_shadow_real.py",
    "voice/test_turn_contract.py",
    "voice/simple.py",
    "voice/simple_daemon.py",
    "voice/simple-daemon",
    "voice/test_simple.py",
    "voice/test_simple_daemon.py",
    "voice/voice_adapter.py",
    "voice/watchdog.py",
    "voice/workspace_router.py",
    "voice/turn_contract.py",
    "contracts/events.py",
    "adapters/codex/app_server.py",
    "adapters/codex/test_app_server.py",
    "adapters/llm/providers.py",
    "adapters/llm/test_providers.py",
    "adapters/voice_pm/publisher.py",
    "adapters/voice_pm/test_publisher.py",
    "adapters/voice_pm/test_switch.py",
    "adapters/voice_pm/test_wiring.py",
    "adapters/voice_pm/wiring.py",
)
LEGACY_RUNTIME_CONTRACT = {
    "candidate_shadow": {
        "microphone": "forbidden",
        "authoritative_thread": "forbidden",
        "audio_sink": "forbidden",
    },
    "production": {
        "selection": "atomic-pointer",
        "source": "verified-bundle-only",
    },
}
MANAGED_RUNTIME_REQUIRED_FILES = (
    "voice/runtime_manager.py",
    "voice/control_plane.py",
    "voice/handoff.py",
    "voice/health.py",
    "voice/release.py",
    "voice/duplex",
    "voice/duplex.py",
    "voice/conversation.py",
    "voice/floor.py",
    "voice/indicator.py",
    "voice/lexicon.json",
    "voice/lexicon.py",
    "voice/modes.py",
    "voice/preflight.py",
    "voice/routes.json",
    "voice/turn_contract.py",
    "voice/voice_adapter.py",
    "voice/workspace_router.py",
    "contracts/events.py",
    "adapters/codex/app_server.py",
    "adapters/llm/providers.py",
    "adapters/voice_pm/publisher.py",
    "adapters/voice_pm/wiring.py",
)
RUNTIME_CONTRACT = {
    **LEGACY_RUNTIME_CONTRACT,
    "production": {
        **LEGACY_RUNTIME_CONTRACT["production"],
        "profile": "managed-runtime-v1",
        "entrypoint": "voice/runtime_manager.py",
        "required_files": list(MANAGED_RUNTIME_REQUIRED_FILES),
        "required_executables": ["voice/duplex"],
    },
}
VERDICT_KEYS = {
    "schema",
    "bundle_sha256",
    "verdict",
    "completed_turns",
    "empty_model_outputs",
    "canary_sha256",
}
FORBIDDEN_CANARY_KEYS = {
    "text",
    "transcript",
    "audio",
    "prompt",
    "messages",
    "content",
    "raw_events",
}


class ReleaseError(ValueError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, relative: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"runtime input must be a regular file: {relative}")
    data = path.read_bytes()
    return {
        "path": relative,
        "sha256": sha256(data),
        "size": len(data),
        "executable": bool(path.stat().st_mode & stat.S_IXUSR),
    }


def manifest_for(
    source: Path,
    allowlist: Iterable[str] = RUNTIME_FILES,
    *,
    runtime_contract: dict[str, Any] = RUNTIME_CONTRACT,
) -> dict[str, Any]:
    source_root = source.resolve()
    files = []
    for relative in sorted(set(allowlist)):
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts:
            raise ReleaseError(f"unsafe runtime allowlist path: {relative}")
        candidate = source / relative
        cursor = source
        for part in parts.parts[:-1]:
            cursor /= part
            if cursor.is_symlink():
                raise ReleaseError(f"runtime input traverses a symlink: {relative}")
        if not candidate.resolve().is_relative_to(source_root):
            raise ReleaseError(f"runtime input escapes source root: {relative}")
        files.append(file_record(candidate, relative))
    body = {
        "schema": SCHEMA,
        "files": files,
        "runtime_contract": runtime_contract,
    }
    return {**body, "bundle_sha256": sha256(canonical(body))}


def bundle_path(state: Path, digest: str) -> Path:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ReleaseError("invalid bundle digest")
    return state / "bundles" / digest


def verify_bundle(path: Path) -> dict[str, Any]:
    manifest_path = path / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ReleaseError("bundle manifest is missing or not regular")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema", "files", "runtime_contract", "bundle_sha256"}:
        raise ReleaseError("bundle manifest fields are invalid")
    body = {
        "schema": manifest["schema"],
        "files": manifest["files"],
        "runtime_contract": manifest["runtime_contract"],
    }
    digest = sha256(canonical(body))
    if manifest["schema"] != SCHEMA or manifest["bundle_sha256"] != digest:
        raise ReleaseError("bundle manifest digest mismatch")
    if path.name != digest:
        raise ReleaseError("bundle directory does not match manifest digest")
    if manifest["runtime_contract"] not in (
        LEGACY_RUNTIME_CONTRACT,
        RUNTIME_CONTRACT,
    ):
        raise ReleaseError("bundle runtime contract mismatch")

    expected = {record["path"]: record for record in manifest["files"]}
    expected_paths = {MANIFEST, *expected}
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_paths.add(str(parent))
            parent = parent.parent
    entries = list(path.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise ReleaseError("bundle contains a symlink")
    actual_paths = {entry.relative_to(path).as_posix() for entry in entries}
    if actual_paths != expected_paths:
        raise ReleaseError("bundle file allowlist mismatch")
    for relative, record in expected.items():
        file = path / relative
        observed = file_record(file, relative)
        if observed != record:
            raise ReleaseError(f"bundle file integrity mismatch: {relative}")
    return manifest


def verify_managed_production(path: Path) -> dict[str, Any]:
    """Require the immutable bundle closure consumed by voice/service."""
    manifest = verify_bundle(path)
    if manifest["runtime_contract"] != RUNTIME_CONTRACT:
        raise ReleaseError(
            "bundle lacks managed-production profile managed-runtime-v1"
        )
    records = {record["path"]: record for record in manifest["files"]}
    missing = [
        relative
        for relative in MANAGED_RUNTIME_REQUIRED_FILES
        if relative not in records
    ]
    if missing:
        raise ReleaseError(
            "managed-production bundle is missing required file: "
            + missing[0]
        )
    for relative in RUNTIME_CONTRACT["production"]["required_executables"]:
        if records[relative].get("executable") is not True:
            raise ReleaseError(
                "managed-production executable bit is missing: " + relative
            )
    return manifest


def stage(
    source: Path,
    state: Path,
    *,
    apply: bool = False,
    allowlist: Iterable[str] = RUNTIME_FILES,
    runtime_contract: dict[str, Any] = RUNTIME_CONTRACT,
) -> dict[str, Any]:
    manifest = manifest_for(
        source,
        allowlist,
        runtime_contract=runtime_contract,
    )
    destination = bundle_path(state, manifest["bundle_sha256"])
    result = {
        "action": "stage",
        "applied": apply,
        "bundle_sha256": manifest["bundle_sha256"],
        "path": str(destination),
        "files": len(manifest["files"]),
    }
    if not apply:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_bundle(destination)
        return result
    temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=destination.parent))
    try:
        for record in manifest["files"]:
            source_file = source / record["path"]
            target = temporary / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
            target.chmod(0o555 if record["executable"] else 0o444)
        (temporary / MANIFEST).write_bytes(canonical(manifest) + b"\n")
        (temporary / MANIFEST).chmod(0o444)
        for record in manifest["files"]:
            if file_record(temporary / record["path"], record["path"]) != record:
                raise ReleaseError(
                    f"runtime input changed while staging: {record['path']}"
                )
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)
    verify_bundle(destination)
    return result


def reject_sensitive_keys(value: object) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_CANARY_KEYS.intersection(map(str, value))
        if forbidden:
            raise ReleaseError(
                f"canary contains transcript-bearing fields: {sorted(forbidden)}"
            )
        for child in value.values():
            reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_sensitive_keys(child)


def verdict_from_canary(canary: dict[str, Any], digest: str) -> dict[str, Any]:
    reject_sensitive_keys(canary)
    promotion = canary.get("promotion")
    if not isinstance(promotion, dict):
        raise ReleaseError("canary promotion verdict is missing")
    verdict = promotion.get("verdict")
    turns = promotion.get("observed_completed_turns")
    if verdict not in {"promote", "hold", "reject"} or not isinstance(turns, int):
        raise ReleaseError("canary promotion verdict is invalid")
    counts = canary.get("counts")
    empty_outputs = (
        counts.get("empty_model_outputs")
        if isinstance(counts, dict)
        else None
    )
    if verdict == "promote" and empty_outputs != 0:
        raise ReleaseError("canary must prove zero empty model outputs")
    if verdict == "promote" and (
        canary.get("status") != "passed"
        or promotion.get("eligible") is not True
    ):
        raise ReleaseError("promote verdict requires a passed, eligible canary")
    if verdict == "promote" and canary.get("pipeline") != PRODUCTION_PIPELINE:
        raise ReleaseError(
            "promotion requires the physical continuous-voice pipeline"
        )
    if verdict == "promote" and canary.get("bundle_sha256") != digest:
        raise ReleaseError("physical canary is bound to another bundle")
    return {
        "schema": SCHEMA,
        "bundle_sha256": digest,
        "verdict": verdict,
        "completed_turns": turns,
        "empty_model_outputs": empty_outputs,
        "canary_sha256": sha256(canonical(canary)),
    }


def validate_verdict(verdict: dict[str, Any], digest: str) -> tuple[str, int]:
    if set(verdict) != VERDICT_KEYS or verdict.get("schema") != SCHEMA:
        raise ReleaseError("release verdict must use the transcript-free schema")
    if verdict.get("bundle_sha256") != digest:
        raise ReleaseError("release verdict is bound to another bundle")
    canary_digest = verdict.get("canary_sha256")
    if (
        not isinstance(canary_digest, str)
        or len(canary_digest) != 64
        or any(character not in "0123456789abcdef" for character in canary_digest)
    ):
        raise ReleaseError("release verdict canary digest is invalid")
    decision = verdict.get("verdict")
    turns = verdict.get("completed_turns")
    empty_outputs = verdict.get("empty_model_outputs")
    if decision not in {"promote", "hold", "reject"} or not isinstance(turns, int):
        raise ReleaseError("release verdict values are invalid")
    if decision == "promote" and empty_outputs != 0:
        raise ReleaseError("promotion requires zero empty model outputs")
    if empty_outputs is not None and not isinstance(empty_outputs, int):
        raise ReleaseError("empty model output count is invalid")
    return decision, turns


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_pointer(state: Path) -> dict[str, Any] | None:
    path = state / "production.json"
    if not path.exists():
        return None
    pointer = json.loads(path.read_text(encoding="utf-8"))
    if set(pointer) != {
        "schema",
        "bundle_sha256",
        "previous_bundle_sha256",
        "release_sha256",
        "generation",
    } or pointer.get("schema") != SCHEMA:
        raise ReleaseError("production pointer is invalid")
    return pointer


@contextmanager
def pointer_lock(state: Path):
    """Serialize pointer reads and mutations across all release processes."""
    state.mkdir(parents=True, exist_ok=True)
    path = state / ".production.lock"
    with path.open("a", encoding="utf-8") as lock:
        os.chmod(path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def release_record_path(state: Path, digest: str) -> Path:
    return state / "verified" / f"{digest}.json"


def verify_release_record(state: Path, digest: str) -> dict[str, Any]:
    path = release_record_path(state, digest)
    if not path.is_file() or path.is_symlink():
        raise ReleaseError(f"bundle has no immutable verification record: {digest}")
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {
        "schema": record.get("schema"),
        "bundle_sha256": record.get("bundle_sha256"),
        "canary_sha256": record.get("canary_sha256"),
        "completed_turns": record.get("completed_turns"),
        "empty_model_outputs": record.get("empty_model_outputs"),
    }
    if (
        set(record) != {*body, "release_sha256"}
        or body["schema"] != SCHEMA
        or body["bundle_sha256"] != digest
        or record["release_sha256"] != sha256(canonical(body))
    ):
        raise ReleaseError("immutable verification record is invalid")
    return record


def write_release_record(
    state: Path, digest: str, verdict: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "bundle_sha256": digest,
        "canary_sha256": verdict["canary_sha256"],
        "completed_turns": verdict["completed_turns"],
        "empty_model_outputs": verdict["empty_model_outputs"],
    }
    record = {**body, "release_sha256": sha256(canonical(body))}
    path = release_record_path(state, digest)
    if path.exists():
        existing = verify_release_record(state, digest)
        if existing != record:
            raise ReleaseError("bundle already has a different verification record")
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, record)
    path.chmod(0o444)
    return record


def _promote_locked(
    state: Path,
    digest: str,
    verdict: dict[str, Any],
    *,
    apply: bool = False,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    verify_managed_production(bundle_path(state, digest))
    decision, turns = validate_verdict(verdict, digest)
    current = read_pointer(state)
    if current:
        resolve_verified_production(state)
    observed_generation = int(current.get("generation", 0)) if current else 0
    if (
        expected_generation is not None
        and expected_generation != observed_generation
    ):
        raise ReleaseError(
            f"production generation changed: expected {expected_generation}, "
            f"observed {observed_generation}"
        )
    before = current.get("bundle_sha256") if current else None
    if decision != "promote":
        return {
            "action": "promote",
            "applied": False,
            "verdict": decision,
            "completed_turns": turns,
            "before": before,
            "after": before,
        }
    if turns < MINIMUM_TURNS:
        raise ReleaseError(f"promotion requires at least {MINIMUM_TURNS} completed turns")
    if before == digest:
        return {
            "action": "promote",
            "applied": False,
            "verdict": decision,
            "completed_turns": turns,
            "before": before,
            "after": before,
        }
    generation = int(current.get("generation", 0)) + 1 if current else 1
    record_body = {
        "schema": SCHEMA,
        "bundle_sha256": digest,
        "canary_sha256": verdict["canary_sha256"],
        "completed_turns": turns,
        "empty_model_outputs": verdict["empty_model_outputs"],
    }
    release_digest = sha256(canonical(record_body))
    pointer = {
        "schema": SCHEMA,
        "bundle_sha256": digest,
        "previous_bundle_sha256": before,
        "release_sha256": release_digest,
        "generation": generation,
    }
    if apply:
        record = write_release_record(state, digest, verdict)
        pointer["release_sha256"] = record["release_sha256"]
        atomic_json(state / "production.json", pointer)
    return {
        "action": "promote",
        "applied": apply,
        "verdict": decision,
        "completed_turns": turns,
        "before": before,
        "after": digest,
        "pointer": pointer,
    }


def promote(
    state: Path,
    digest: str,
    verdict: dict[str, Any],
    *,
    apply: bool = False,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    with pointer_lock(state):
        return _promote_locked(
            state,
            digest,
            verdict,
            apply=apply,
            expected_generation=expected_generation,
        )


def resolve_verified_production(state: Path) -> Path:
    """Resolve the current pointer using immutable integrity only.

    Promotion uses this during the one-way migration from a verified legacy
    emergency bundle to the managed production profile.
    """
    pointer = read_pointer(state)
    if not pointer:
        raise ReleaseError("no production voice bundle is pinned")
    digest = pointer["bundle_sha256"]
    path = bundle_path(state, digest)
    verify_bundle(path)
    record = verify_release_record(state, digest)
    if record["release_sha256"] != pointer["release_sha256"]:
        raise ReleaseError("production pointer verification mismatch")
    return path


def resolve_production(state: Path) -> Path:
    path = resolve_verified_production(state)
    verify_managed_production(path)
    return path


def release_status(state: Path) -> dict[str, Any]:
    """Report a verified pointer even while migrating a legacy production."""
    pointer = read_pointer(state)
    if not pointer:
        return {
            "production": None,
            "pointer": None,
            "managed_compatible": False,
            "managed_compatibility_error": "no production voice bundle is pinned",
        }
    path = resolve_verified_production(state)
    try:
        verify_managed_production(path)
    except ReleaseError as error:
        compatible = False
        reason: str | None = str(error)
    else:
        compatible = True
        reason = None
    return {
        "production": str(path),
        "pointer": pointer,
        "managed_compatible": compatible,
        "managed_compatibility_error": reason,
    }


def _rollback_locked(
    state: Path,
    *,
    apply: bool = False,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    current = read_pointer(state)
    if not current or not current.get("previous_bundle_sha256"):
        raise ReleaseError("no prior verified voice bundle is available")
    observed_generation = int(current["generation"])
    if (
        expected_generation is not None
        and expected_generation != observed_generation
    ):
        raise ReleaseError(
            f"production generation changed: expected {expected_generation}, "
            f"observed {observed_generation}"
        )
    resolve_verified_production(state)
    before = current["bundle_sha256"]
    previous = current["previous_bundle_sha256"]
    verify_managed_production(bundle_path(state, previous))
    record = verify_release_record(state, previous)
    pointer = {
        "schema": SCHEMA,
        "bundle_sha256": previous,
        "previous_bundle_sha256": before,
        "release_sha256": record["release_sha256"],
        "generation": int(current["generation"]) + 1,
    }
    if apply:
        atomic_json(state / "production.json", pointer)
    return {
        "action": "rollback",
        "applied": apply,
        "before": before,
        "after": previous,
        "pointer": pointer,
    }


def rollback(
    state: Path,
    *,
    apply: bool = False,
    expected_generation: int | None = None,
) -> dict[str, Any]:
    with pointer_lock(state):
        return _rollback_locked(
            state,
            apply=apply,
            expected_generation=expected_generation,
        )


def default_state() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return base / "zer0-voice/releases"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=default_state())
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--source", type=Path, default=ROOT)
    stage_parser.add_argument("--apply", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("bundle")
    verdict_parser = subparsers.add_parser("verdict")
    verdict_parser.add_argument("bundle")
    verdict_parser.add_argument("--canary", type=Path, required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("bundle")
    promote_parser.add_argument("--verdict", type=Path, required=True)
    promote_parser.add_argument("--apply", action="store_true")
    promote_parser.add_argument("--expected-generation", type=int)
    subparsers.add_parser("status")
    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--path", action="store_true")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--apply", action="store_true")
    rollback_parser.add_argument("--expected-generation", type=int)
    args = parser.parse_args()
    try:
        if args.command == "stage":
            result = stage(args.source, args.state, apply=args.apply)
        elif args.command == "verify":
            result = verify_bundle(bundle_path(args.state, args.bundle))
        elif args.command == "verdict":
            canary = json.loads(args.canary.read_text(encoding="utf-8"))
            result = verdict_from_canary(canary, args.bundle)
        elif args.command == "promote":
            if args.apply and args.expected_generation is None:
                raise ReleaseError(
                    "--apply requires --expected-generation from release status"
                )
            verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
            result = promote(
                args.state,
                args.bundle,
                verdict,
                apply=args.apply,
                expected_generation=args.expected_generation,
            )
        elif args.command == "rollback":
            if args.apply and args.expected_generation is None:
                raise ReleaseError(
                    "--apply requires --expected-generation from release status"
                )
            result = rollback(
                args.state,
                apply=args.apply,
                expected_generation=args.expected_generation,
            )
        elif args.command == "resolve":
            path = resolve_production(args.state)
            if args.path:
                print(path)
                return 0
            result = {"production": str(path), "pointer": read_pointer(args.state)}
        else:
            result = release_status(args.state)
    except (OSError, ReleaseError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
