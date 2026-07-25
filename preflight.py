#!/usr/bin/env python3
"""Fail-fast dependency and routing checks for continuous local voice."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from voice_adapter import asr_health, tts_health
from workspace_router import WorkspaceRouter, load_routes


def request_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def matching_device(devices: list[dict], query: str | None, direction: str) -> dict | None:
    channel = "max_input_channels" if direction == "input" else "max_output_channels"
    candidates = [device for device in devices if device.get(channel, 0) > 0]
    if query is None:
        return next((device for device in candidates if device.get("default")), None)
    lowered = query.lower()
    return next(
        (device for device in candidates if lowered in str(device.get("name", "")).lower()),
        None,
    )


def parse_pulse_sinks(output: str) -> list[str]:
    return [
        fields[1]
        for line in output.splitlines()
        if len(fields := line.split("\t")) >= 2
    ]


def preflight(
    *,
    whisper_python: str,
    kokoro_url: str,
    ollama_url: str | None,
    live_model: str | None,
    input_device: str | None,
    output_device: str | None,
    workspace_routing: bool,
    routes: Path,
    require_input: bool = True,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    failures: list[str] = []
    warnings: list[str] = []

    checks["asr"] = asr_health(whisper_python)
    if not checks["asr"]["ok"]:
        failures.append("Faster Whisper environment is unavailable")
    else:
        cuda = subprocess.run(
            [
                whisper_python,
                "-c",
                "import ctranslate2; print(ctranslate2.get_cuda_device_count())",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        cuda_devices = int(cuda.stdout.strip() or 0) if cuda.returncode == 0 else 0
        checks["asr"]["cuda_devices"] = cuda_devices
        if cuda_devices < 1:
            failures.append("Faster Whisper cannot see a CUDA device")

    checks["tts"] = tts_health(kokoro_url)
    if not checks["tts"]["ok"]:
        failures.append("Kokoro is not healthy")

    if ollama_url and live_model:
        try:
            tags = request_json(f"{ollama_url.rstrip('/')}/api/tags")
            models = [model.get("name") for model in tags.get("models", ())]
            checks["live_model"] = {
                "ok": live_model in models,
                "url": ollama_url,
                "model": live_model,
                "available": models,
            }
            if live_model not in models:
                failures.append(f"Ollama model {live_model!r} is not installed")
            processes = request_json(f"{ollama_url.rstrip('/')}/api/ps")
            loaded = next(
                (
                    model
                    for model in processes.get("models", ())
                    if model.get("name") == live_model
                ),
                None,
            )
            if loaded and not loaded.get("size_vram"):
                warnings.append(f"Ollama model {live_model!r} is CPU-only")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            checks["live_model"] = {"ok": False, "url": ollama_url, "error": str(exc)}
            failures.append("Ollama is unreachable")

    if not require_input:
        checks["input"] = {
            "ok": True,
            "requested": input_device,
            "selected": None,
            "deferred": True,
        }
    else:
        try:
            import sounddevice as sd

            devices = [dict(device) for device in sd.query_devices()]
            selected_input = matching_device(devices, input_device, "input")
            checks["input"] = {
                "ok": selected_input is not None,
                "requested": input_device,
                "selected": selected_input.get("name") if selected_input else None,
            }
            if selected_input is None:
                failures.append(f"input device {input_device!r} is unavailable")
        except Exception as exc:
            checks["input"] = {"ok": False, "error": str(exc)}
            failures.append("audio input discovery failed")

    player = shutil.which("pw-play")
    sinks: list[str] = []
    if player and shutil.which("pactl"):
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            sinks = parse_pulse_sinks(result.stdout)
    output_ok = player is not None and (
        output_device is None or output_device in sinks
    )
    checks["output"] = {
        "ok": output_ok,
        "player": player,
        "requested": output_device,
        "available": sinks,
    }
    if player is None:
        failures.append("pw-play is unavailable")
    elif output_device and output_device not in sinks:
        failures.append(f"PipeWire output {output_device!r} is unavailable")

    if workspace_routing:
        try:
            resolution = WorkspaceRouter(load_routes(routes)).resolve()
            checks["routing"] = {
                "ok": True,
                "reason": resolution.reason,
                "route": resolution.route.project if resolution.route else None,
                "candidates": [route.project for route in resolution.candidates],
            }
            if resolution.route is None:
                warnings.append(
                    f"workspace routing is {resolution.reason}; launch context will be used"
                )
        except Exception as exc:
            checks["routing"] = {"ok": False, "error": str(exc)}
            warnings.append("workspace routing preflight failed; launch context will be used")

    return {
        "schema": 1,
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-python", default="/workspace/whisper/.venv/bin/python")
    parser.add_argument("--kokoro-url", default="http://127.0.0.1:8880")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--live-model", default="qwen2.5:3b")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--workspace-routing", action="store_true")
    parser.add_argument("--routes", type=Path, default=Path(__file__).parent / "routes.json")
    args = parser.parse_args()
    result = preflight(
        whisper_python=args.whisper_python,
        kokoro_url=args.kokoro_url,
        ollama_url=args.ollama_url,
        live_model=args.live_model,
        input_device=args.input,
        output_device=args.output,
        workspace_routing=args.workspace_routing,
        routes=args.routes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
