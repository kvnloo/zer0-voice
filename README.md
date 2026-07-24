# Zer0 voice adapters

This directory bridges the harness to the existing machine-wide voice services
without copying their models or Python dependencies into this project.

```sh
python voice/voice_adapter.py health
python voice/voice_adapter.py synthesize "Zer0 voice is connected." \
  --output /tmp/zer0-voice.wav
```

Defaults:

- Faster Whisper: `/workspace/whisper/.venv/bin/python`
- Kokoro OpenAI-compatible API: `http://127.0.0.1:8880`

Override them with `ZERO_WHISPER_PYTHON` and `ZERO_KOKORO_URL`, or the matching
CLI flags. Start the existing Kokoro service from `/workspace/kokoro-tts` with
its `start-cpu.sh`; the adapter deliberately does not own that global service.

`health` verifies that Faster Whisper imports in its actual venv and that the
Kokoro model server has completed startup. `synthesize` calls
`POST /v1/audio/speech` and writes the returned WAV.

Run the dependency-free adapter tests with:

```sh
python -m unittest discover -s voice -p 'test_*.py'
```

## Continuous Codex conversation

`conversation` is a hands-free loop that uses the existing Codex subscription
and resumes one persistent Codex session. Audio never needs to leave the
machine: Faster Whisper handles recognition and Kokoro handles speech.

Start the Kokoro service, then run:

```sh
./voice/conversation --cwd /workspace/zer0/products/pm
```

By default it resumes the newest Codex conversation. Pin a thread when several
Codex sessions are active:

```sh
./voice/conversation --session SESSION_UUID
```

The loop uses energy-based endpoint detection, cached `small.en` Whisper on
CPU, concise Codex responses, and the Kokoro `af_heart` voice. Tune a noisy or
quiet microphone with `--threshold`; use `--silence-ms` to change how quickly a
turn ends. Run `--once` for a single voice round trip.

`floor.py` contains the deterministic full-duplex floor policy. It distinguishes
backchannels from real interruptions, ducks immediately while classifying
overlap, yields for corrections/questions, and can briefly hold the floor for an
important or safety-critical thought. Its adaptive endpoint tolerates reflective
pauses instead of treating every silence as the end of a turn.

The simple `conversation` executable still speaks after a Codex turn completes.
`duplex` is the production local loop: it keeps the microphone open while Codex
and Kokoro speak, streams authenticated Codex app-server deltas into sentence
sized Kokoro chunks, stops output on speech-start, interrupts the active Codex
turn, endpoints the user's interruption, and immediately continues on the same
thread.

```sh
./voice/duplex --cwd /workspace/zer0/products/pm
```

Before loading models or opening the microphone, `duplex` runs a read-only
preflight over the Whisper CUDA environment, Kokoro health, installed Ollama
model, requested audio devices, `pw-play`, and privacy-safe workspace routing.
Failures stop immediately with structured diagnostics; warnings identify
non-fatal conditions such as CPU-only Ollama or ambiguous tmux focus. Use
`--skip-preflight` only when an external supervisor has already run the same
checks.

The launcher supplies the CUDA 12 libraries already installed with the global
Kokoro environment, allowing Faster Whisper `small.en` to run on the GPU without
duplicating those packages.

By default, each completed utterance resolves the privacy-filtered
`workspace-copilot` context and snapshots its focused tmux harness as the owner
of that turn. Switching windows does not redirect an answer that is already
speaking. If the user interrupts, the pending utterance resolves focus again and
can route to the newly focused harness. Each routed project has isolated live
history and deep insights, and its newest interactive Codex thread is resumed
when available.

Project paths are configured in `voice/routes.json`. Disable routing with
`--no-workspace-routing`, or pin the deep lane with `--session THREAD_ID`.
Routing fails closed to the launch-context thread when the sanitized workspace
sensor reports multiple active harness windows without a focused tmux pane.
Full window-to-window switching therefore requires `workspace-copilot` context
to expose either `tmux_focus.pane_id` or an `active` bit on the displayed tmux
window; raw pane contents and window titles are never inspected.

When focus is ambiguous, routing remains hands-free: say `switch to zerOS`,
`talk to PM`, `route to Flowkit`, or `talk to dotfiles` to pin subsequent
turns. Say `follow focus` to return to automatic workspace routing.

## Performance ledger

Pass `--metrics bench/voice-history.jsonl` to append one privacy-safe latency
record per turn. Records contain routing disposition and stage timings, never
transcript text or audio. Verify the current ledger with:

```sh
python voice/verify_metrics.py
```

The initial median budgets are 500 ms for ASR, 500 ms for first TTS synthesis,
2.5 seconds from endpoint to first audio, and 3 seconds estimated from the
user's last voiced block to first audio. Interrupted turns remain visible in
the ledger but do not poison latency comparisons.

`fleet.py` implements the parallel intelligence hierarchy. The default profile
is five total lanes: one live voice generator plus instant, medium, high, and
pro reasoning lanes. Instant and medium run every turn; high and pro use a
cadence unless a material event wakes all lanes. Results stream in completion
order and carry horizon, latency, deadline, and intervention severity metadata,
so a fast correction can steer speech without waiting for meta-strategy.

The lanes form a mesh rather than a command hierarchy. `blackboard.py` gives
every lane the same causal turn board, lets any lane supersede another lane's
proposal, ranks urgent/confident interventions, and bounds deliberation rounds
to prevent recursive agent chatter. The live voice lane consumes this board
while speaking, so smarter results can revise it without becoming a latency
gate.
