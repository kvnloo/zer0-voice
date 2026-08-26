# xAI Grok realtime voice plugin

This directory is a Hermes plugin for the provider/session seam introduced by Hermes PR #95147 at `3f33ac9daff8b903d237613aaf61caa03a6e83e4`. It is a transport only: Hermes remains the sole owner of conversation history, tools, approvals, memory, and answer authority. Existing zer0-voice Codex/Whisper/Kokoro paths are unchanged and remain the fallback when this optional plugin is disabled or unavailable.

Install or symlink this directory as `$HERMES_HOME/plugins/xai-realtime-voice`, then enable `xai-realtime-voice` under `plugins.enabled` in the active profile's `config.yaml`. The target Hermes runtime must include its documented realtime voice contract and the `websockets` package.

Server-side environment only:

- `XAI_API_KEY` (required; never send it to a browser)
- `HERMES_XAI_VOICE_SPEED` (optional, `0.7`–`1.5`, default `1.0`)
- `HERMES_XAI_VOICE_REASONING` (optional, `high` or `none`, default `high`)

The provider opens `grok-voice-latest`, enables resumable sessions, supplies only Hermes function schemas (no xAI web/X/file/MCP server tools), streams corrected cumulative user transcripts and PCM output deltas, groups parallel function results before one continuation, and truncates provider history to the host-reported heard-audio boundary on barge-in. Metrics are counters only.

Verification:

    python -m unittest test_xai_realtime_voice -v
    python -m unittest discover -s . -p 'test_*.py'

Protocol reference inspected during implementation: https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech.md
