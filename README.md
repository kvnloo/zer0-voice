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
The production loop will drive the same adapters through Codex app-server
message deltas, streaming Kokoro playback, echo cancellation, and the floor
policy rather than waiting for completed turns.

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
