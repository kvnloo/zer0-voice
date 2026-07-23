#!/usr/bin/env python3
"""Small, dependency-free bridge to the machine's Whisper and Kokoro installs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WHISPER_PYTHON = os.getenv(
    "ZERO_WHISPER_PYTHON", "/workspace/whisper/.venv/bin/python"
)
KOKORO_URL = os.getenv("ZERO_KOKORO_URL", "http://127.0.0.1:8880")


def request_json(url: str, *, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def asr_health(python: str = WHISPER_PYTHON, timeout: float = 10.0) -> dict[str, Any]:
    executable = Path(python)
    if not executable.is_file():
        return {"ok": False, "kind": "asr", "error": "python_not_found", "python": python}
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import faster_whisper; print(faster_whisper.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "kind": "asr", "error": "import_timeout", "python": python}
    return {
        "ok": result.returncode == 0,
        "kind": "asr",
        "python": python,
        "engine": "faster-whisper",
        "version": result.stdout.strip() or None,
        "error": result.stderr.strip() or None,
    }


def tts_health(base_url: str = KOKORO_URL, timeout: float = 2.0) -> dict[str, Any]:
    try:
        body = request_json(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return {
            "ok": body.get("status") == "healthy",
            "kind": "tts",
            "engine": "kokoro",
            "url": base_url,
            "status": body.get("status"),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "kind": "tts",
            "engine": "kokoro",
            "url": base_url,
            "error": str(exc),
        }


def synthesize(
    text: str,
    output: Path,
    *,
    base_url: str = KOKORO_URL,
    voice: str = "af_heart",
    model: str = "kokoro",
    timeout: float = 120.0,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        return {"ok": False, "kind": "tts", "error": f"HTTP {exc.code}: {detail}"}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "kind": "tts", "error": str(exc)}
    output.write_bytes(audio)
    return {
        "ok": len(audio) > 44,
        "kind": "tts",
        "output": str(output),
        "bytes": len(audio),
        "voice": voice,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health", help="check ASR import and Kokoro HTTP health")
    health.add_argument("--whisper-python", default=WHISPER_PYTHON)
    health.add_argument("--kokoro-url", default=KOKORO_URL)
    speech = commands.add_parser("synthesize", help="write Kokoro speech to a WAV file")
    speech.add_argument("text")
    speech.add_argument("--output", type=Path, default=Path("speech.wav"))
    speech.add_argument("--voice", default="af_heart")
    speech.add_argument("--model", default="kokoro")
    speech.add_argument("--kokoro-url", default=KOKORO_URL)
    args = parser.parse_args()

    if args.command == "health":
        result: dict[str, Any] = {
            "ok": False,
            "asr": asr_health(args.whisper_python),
            "tts": tts_health(args.kokoro_url),
        }
        result["ok"] = result["asr"]["ok"] and result["tts"]["ok"]
    else:
        result = synthesize(
            args.text,
            args.output,
            base_url=args.kokoro_url,
            voice=args.voice,
            model=args.model,
        )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
