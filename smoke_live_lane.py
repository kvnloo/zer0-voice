#!/usr/bin/env python3
"""Bounded real app-server smoke test for the ephemeral realtime lane."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [
    str(HERE.parents[1] / "contracts"),
    str(HERE.parents[1] / "adapters/codex"),
    str(HERE.parents[1] / "adapters/llm"),
]

from app_server import CodexAppServer
from duplex import LIVE_LANE_INSTRUCTIONS
from providers import CodexSubscription


async def smoke(cwd: Path, timeout: float) -> dict[str, object]:
    async with CodexAppServer(cwd=cwd, shared=True) as server:
        thread = await server.start_thread(
            cwd=cwd,
            developer_instructions=LIVE_LANE_INSTRUCTIONS,
            ephemeral=True,
        )
        events = []

        async def collect() -> None:
            async for event in CodexSubscription(server, thread).stream(
                "Reply with one short natural sentence confirming the live lane.",
                effort="low",
            ):
                events.append(
                    {
                        "kind": event.kind,
                        "text_bytes": len(event.text.encode()),
                    }
                )

        await asyncio.wait_for(collect(), timeout=timeout)
        return {
            "schema": 1,
            "ok": any(event["kind"] == "completed" for event in events),
            "events": events,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    result = asyncio.run(smoke(args.cwd, args.timeout))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
