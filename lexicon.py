"""Project-aware ASR vocabulary and conservative phrase correction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TERMS = (
    "Zer0",
    "Zer0 PM",
    "ZerOS",
    "Codex",
    "Codex harness",
    "OMP",
    "tmux",
    "tldraw",
    "XYFlow",
    "Kokoro",
    "TTS",
    "ASR",
    "Whisper",
    "PipeWire",
    "Ratatui",
    "Rust",
    "Go",
    "C++",
    "Python",
    "TypeScript",
    "React",
    "WebSocket",
    "MCP",
    "LLM",
    "VAD",
    "CUDA",
    "PyTorch",
    "JAX",
    "llama.cpp",
    "Ollama",
    "Hugging Face",
    "DeepSeek",
    "Qwen",
    "Kimi",
    "MiniMax",
    "GLM",
    "Sol",
    "Terra",
    "minimal effort",
    "low effort",
    "medium effort",
    "high effort",
    "max effort",
    "agentic coding",
    "meta-orchestration",
)

# Only correct phrases that are highly unlikely to be intentional in this
# technical context. Ambiguous words such as "flow" or "go" are hotwords only.
DEFAULT_CORRECTIONS = (
    (r"\bOh My Pie\b", "OMP"),
    (r"\bTeamux\b", "tmux"),
    (r"\bT Mux\b", "tmux"),
    (r"\bCocoro\b", "Kokoro"),
    (r"\bCocoa Ro\b", "Kokoro"),
    (r"\bT(?:\s*)T(?:\s*)A\b", "TTS"),
    (r"\bX Y Flow\b", "XYFlow"),
    (r"\bT L Draw\b", "tldraw"),
    (r"\bZero O S\b", "ZerOS"),
)


@dataclass(frozen=True)
class Lexicon:
    terms: tuple[str, ...] = DEFAULT_TERMS
    corrections: tuple[tuple[str, str], ...] = DEFAULT_CORRECTIONS

    @property
    def hotwords(self) -> str:
        return ", ".join(self.terms)

    @property
    def prompt(self) -> str:
        # `faster-whisper` already injects `hotwords` into the decoder prompt.
        # Keep the initial prompt contextual but vocabulary-free so silence and
        # coughs are not biased toward a second copy of terms such as "JAX".
        return "Technical conversation about AI coding, local voice, and project management."

    def correct(self, text: str) -> str:
        for pattern, replacement in self.corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


def load_lexicon(*paths: Path) -> Lexicon:
    terms = list(DEFAULT_TERMS)
    corrections = list(DEFAULT_CORRECTIONS)
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        terms.extend(str(term) for term in value.get("terms", []))
        corrections.extend(
            (str(item["pattern"]), str(item["replacement"]))
            for item in value.get("corrections", [])
        )
    # Preserve priority/order while removing duplicate hotwords.
    return Lexicon(tuple(dict.fromkeys(terms)), tuple(corrections))
