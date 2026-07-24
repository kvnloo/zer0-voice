#!/usr/bin/env python3
"""Privacy-safe proof that a stable harness-history fork can answer live."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "adapters/codex"), str(ROOT / "contracts")]

from app_server import CodexAppServer


async def verify(session: str, cwd: Path, timeout: float) -> dict[str, object]:
    async with CodexAppServer(cwd=cwd, shared=True) as server:
        authoritative = await server.resume_thread(session, cwd=cwd)
        source = await server.read_thread(authoritative)
        turns = source.get("turns", [])
        last_completed = next(
            (
                str(turn["id"])
                for turn in reversed(turns)
                if turn.get("status") == "completed" and turn.get("id")
            ),
            None,
        )
        fork = await server.fork_thread(
            authoritative,
            cwd=cwd,
            developer_instructions=(
                "Realtime voice context verification. Do not start a turn."
            ),
            ephemeral=True,
            last_turn_id=last_completed,
        )
        deltas = 0
        completed = False

        async def collect() -> None:
            nonlocal deltas, completed
            async for event in server.stream_turn(
                fork,
                "Reply with one short sentence confirming realtime context.",
                effort="low",
            ):
                deltas += event.kind == "assistant.delta"
                completed = completed or event.kind == "assistant.completed"

        await asyncio.wait_for(collect(), timeout=timeout)
        return {
            "schema": 1,
            "ok": bool(fork and fork != authoritative and completed and deltas),
            "source_turns": len(turns),
            "source_in_progress": any(
                turn.get("status") == "inProgress" for turn in turns
            ),
            "stable_boundary_found": last_completed is not None,
            "fork_is_distinct": bool(fork and fork != authoritative),
            "live_deltas": deltas,
            "live_completed": completed,
            "ephemeral": True,
            "transcript_content_read": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if not args.session:
        parser.error("--session or CODEX_THREAD_ID is required")
    result = asyncio.run(verify(args.session, args.cwd, args.timeout))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
