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

### Staged simple runtime

`simple.py` is the fail-closed recovery path for continuous conversation. It
snapshots the newest completed turn from the selected harness exactly once,
creates one ephemeral Codex conversation that inherits that history, and reuses
the same conversation for every subsequent voice turn. One `TurnOwner`
determines committed ASR boundaries; one lock serializes model requests; and one
bounded FIFO serializes Kokoro playback. It has no workspace rerouting,
authoritative submission, reasoning fleet, barge-in, or per-turn refork.

The module is staged but is not selected by the production service merely by
editing it. It is included in the immutable release allowlist and must pass its
ten-turn context/ordering cohort before an explicit release promotion. Empty
model output and empty speech are visible errors and never count as completed
turns.

`simple-daemon` is the executable recovery runtime around that core. It is
strictly sequential: capture one complete thought, transcribe it, run one turn
on the single inherited conversation, finish Kokoro/PipeWire playback, then
listen again. It has no restart loop, routing, PM publishing,
authoritative-thread submission, barge-in, reasoning lanes, or watchdog.
Failures exit visibly instead of pretending the service is healthy.

The configuration-only and hardware probes never attach to a Codex thread or
start a conversation:

```sh
./voice/simple-daemon --smoke --cwd "$PWD"
./voice/simple-daemon --hardware-smoke --cwd "$PWD"
```

Run the recovery loop only after its immutable bundle passes the release gate:

```sh
./voice/simple-daemon --session "$CODEX_THREAD_ID" \
  --output effect_input.aural_evolution \
  --health /tmp/zer0-simple-voice-health.json \
  --metrics /tmp/zer0-simple-voice-metrics.jsonl
```

```sh
PYTHONPATH=voice:contracts:adapters/llm:adapters/codex \
  python -m unittest voice.test_simple voice.test_simple_daemon voice.test_release
```

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

The production duplex runtime defaults to `--barge-in final`. Speech capture
continues while a reply is pending, but a completed follow-up is buffered
instead of canceling a reply that has not begun playback. The replaceable live
lane uses `gpt-5.6-luna` at low effort by default; the authoritative harness
still receives every committed turn independently. `sustained` and `immediate`
remain explicit opt-ins for environments where aggressive interruption is more
important than preserving a slow unspoken reply.

Input and proactive speech density are independent runtime modes:

```sh
./voice/control mic continuous
./voice/control mic muted
./voice/control mic push-to-talk
./voice/control press
./voice/control release

./voice/control notify conversational
./voice/control notify updates
./voice/control notify critical
./voice/control status
```

`muted` physically closes the capture stream. Push-to-talk opens it on `press`
and closes/finalizes held audio on `release`; those commands are suitable for a
Hyprland press/release key binding. Notification density never suppresses a
direct answer: `updates` and `critical` filter only proactive/background spoken
events. The private control socket is mode `0600` and is removed on shutdown.

The Wooting Esc status light is deliberately opt-in:

```sh
ZERO_WOOTING_INDICATOR=wooting \
WOOTING_RGB_LIB=/path/to/libwooting-rgb-sdk.so \
  ./voice/codex-harness --voice "$CODEX_THREAD_ID"
```

It uses only the official transient single-key set/reset calls. It does not
write, switch, import, or export keyboard profiles. Cyan means listening, amber
thinking, violet speaking, and red error; muted/shutdown resets only Esc to the
active onboard profile color.

Utterance endpointing defaults to 850 ms of silence. The earlier 520 ms setting
split ordinary reflective speech into unrelated harness turns during a real
conversation. This is intentionally conservative until semantic endpointing
can distinguish a completed thought from a mid-sentence pause.

The simple `conversation` executable still speaks after a Codex turn completes.
`duplex` is the production local loop: it keeps the microphone open while Codex
and Kokoro speak, sends each transcript as a real turn on the selected existing
Codex harness thread, streams that thread's authenticated app-server deltas into
sentence-sized Kokoro chunks, interrupts the active Codex turn on sustained
speech, endpoints the interruption, and continues on the same thread. No local
model generates assistant answers.

```sh
./voice/duplex --cwd /workspace/zer0/products/pm
```

For the transcript and streamed reply to appear live in the same terminal UI,
launch both clients against Codex's managed app-server:

```sh
./voice/codex-harness --voice "$CODEX_THREAD_ID"
```

This resumes the TUI through `codex --remote unix://`, starts `duplex` against
the daemon proxy, pins voice to that exact thread, and stops the microphone when
the TUI exits. The launcher health-checks Kokoro and, when it is offline, starts
the existing global service through `/workspace/kokoro-tts/kokoro.sh`; override
that path with `ZERO_KOKORO_LAUNCHER`. Voice logs go to
`$XDG_STATE_HOME/zer0-voice/THREAD_ID.log` (or
`~/.local/state/zer0-voice/THREAD_ID.log`). `ZERO_VOICE_INPUT` and
`ZERO_VOICE_OUTPUT` override the default Blue Snowball and Aural Evolution
devices. A legacy TUI that was not launched through the shared server cannot
subscribe to turns created by a second app-server process; restart it with this
launcher rather than pretending the detached process is integrated.

When the interactive TUI is already attached, keep the voice stack alive in a
separate tmux service window without launching a second TUI:

```sh
./voice/service "$CODEX_THREAD_ID"
```

Before the next service launch, stage and explicitly promote an immutable
runtime. Every command is a dry run unless `--apply` is supplied:

```sh
python voice/release.py stage
python voice/release.py stage --apply

# First run the isolated GPU/model shadow. It proves adapter quality without
# production mic/thread/sink ownership, but cannot authorize production.
STAGED_BUNDLE/voice/shadow-real \
  --report /tmp/voice-shadow.json

# Then run the physical candidate and collect ten real continuous turns.
# Only voice/canary.py's codex-continuous-pcm-v5 report can authorize release.
PYTHONPATH=voice python voice/canary.py \
  --metrics bench/voice-history.jsonl \
  --debug "${XDG_STATE_HOME:-$HOME/.local/state}/zer0-voice/voice-debug.jsonl" \
  --health "${XDG_STATE_HOME:-$HOME/.local/state}/zer0-voice/health.json" \
  > /tmp/voice-physical-canary.json

python voice/release.py verdict BUNDLE_SHA \
  --canary /tmp/voice-physical-canary.json > /tmp/voice-verdict.json

python voice/release.py promote BUNDLE_SHA \
  --verdict /tmp/voice-verdict.json
python voice/release.py promote BUNDLE_SHA \
  --verdict /tmp/voice-verdict.json --apply
```

Only an exact `promote` verdict with at least ten completed turns can replace
`production.json`; `hold`, `reject`, malformed, transcript-bearing, short, or
bundle-mismatched verdicts leave it byte-for-byte unchanged. Promotion also
requires an explicit privacy-safe `empty_model_outputs: 0` count; missing or
nonzero evidence fails closed. A passing isolated `voice-shadow-v1` report is
necessary adapter evidence but is deliberately rejected as release authority;
the verdict must identify the physical `codex-continuous-pcm-v5` pipeline.
Bundle manifests
hash every allowlisted runtime file, use the manifest hash as the directory
name, reject symlinks and extra files, and are reverified on every resolve.
Promotion writes an immutable verification record before atomically replacing
the production pointer. Rollback is also dry-run first and can select only the
previous verified bundle:

```sh
python voice/release.py rollback
python voice/release.py rollback --apply
```

The release CLI never launches a candidate. Shadow/canary code is contractually
forbidden from owning the production microphone, authoritative Codex thread,
or audio sink. `service` resolves a verified bundle and runs that bundle's
pinned `runtime_manager.py`; dirty worktree Python cannot enter a restart or
rollout. The manager follows later verified pointer changes, warms the selected
immutable generation with capture muted, switches only at an idle boundary,
and retains the prior attached generation through health probation.

`service` holds a single-instance lock, waits for Kokoro's CUDA warmup, starts
the canonical PM event relay when needed, and delegates generations to one
stable manager/control endpoint. A privacy-safe heartbeat records only the
worker PID, run, phase, lane, and timestamps. Stale heartbeats, bounded
attach/transcribe/generate/speak deadlines, process exits, and Kokoro failure
trigger repair and replacement. Failed rollouts restore the old mic owner
before terminating the candidate. Every generation attaches to the same
supplied conversation; reattaching the visible TUI does not create another.

Inspect functional health without reading transcripts:

```sh
PYTHONPATH=voice python voice/health.py \
  "${XDG_STATE_HOME:-$HOME/.local/state}/zer0-voice/health.json"
```

Before loading models or opening the microphone, `duplex` runs a read-only
preflight over the Whisper CUDA environment, Kokoro health, requested audio
devices, `pw-play`, and privacy-safe workspace routing. Failures stop
immediately with structured diagnostics; warnings identify non-fatal conditions
such as ambiguous tmux focus. Use `--skip-preflight` only when an external
supervisor has already run the same checks.

The launcher supplies the CUDA 12 libraries already installed with the global
Kokoro environment, allowing Faster Whisper `small.en` to run on the GPU without
duplicating those packages.

By default, each completed utterance resolves the privacy-filtered
`workspace-copilot` context and snapshots its focused tmux harness as the owner
of that turn. Switching windows does not redirect an answer that is already
speaking. If the user interrupts, the pending utterance resolves focus again and
can route to the newly focused harness. Each routed project has isolated live
thread identity, and its newest interactive Codex thread is resumed when
available.

The launch harness is selected from `--session`, then `CODEX_THREAD_ID`, then
the newest existing interactive thread in `--cwd`. The runtime never silently
creates a detached fallback conversation. A routing switch likewise resumes an
existing interactive thread for that project or fails closed.

The bridge does not copy recent messages into a second prompt. Resuming the
actual thread gives Codex its authoritative conversation and tool history
without a redundant context fetch or another model.

Project paths are configured in `voice/routes.json`. Disable routing with
`--no-workspace-routing`, or pin the harness with `--session THREAD_ID`.
`workspace-copilot` now emits `tmux_focus` (pane id, session, window index of
the most recently active attached client) and an `active` bit on the current
window of each attached session; raw pane contents and window titles are never
inspected. Resolution order per utterance: the focused pane when it is itself a
harness, then the single harness in the focused window (pane activity breaks a
tie between several harnesses there), then the unique harness across active
windows, then the unique routed harness overall. A harness pane no longer needs
to be its window's active pane — focusing a dashboard or build pane beside the
harness still attaches to that window's harness. If the focused window holds
several equally plausible harnesses, routing fails closed to the launch-context
thread rather than rerouting to a background window.

When focus is ambiguous, routing remains hands-free: say `switch to zerOS`,
`talk to PM`, `route to Flowkit`, or `talk to dotfiles` to pin subsequent
turns. Say `follow focus` to return to automatic workspace routing.

## Hermetic shadow candidates

`shadow.py` evaluates candidate ASR/model/TTS changes without attaching to the
production microphone, authoritative Codex thread, or an audio device. Callers
must inject an explicit prerecorded or synthetic `ShadowCase` corpus, an ASR
adapter marked `explicit-corpus`, a model adapter marked
`isolated-ephemeral`, a PCM synthesizer, and a `discard-hash` sink. The built-in
sink hashes PCM in memory and discards it.

A case may also carry an explicit synthetic or prerecorded
`completed_turn_snapshot` byte payload. It is passed only to the isolated model
adapter, allowing a fork-from-completed-turn candidate path to be exercised
without a production thread ID, app-server attachment, or authoritative state.

The result contains only case IDs constrained to a safe character set, stage
order, counts, latency summaries, PCM byte counts/hashes, and promotion state.
Transcript, prompt, response, content, and adapter exception text are forbidden
from the report schema. Timeouts, adapter errors, duplicate finals, unsafe
adapter modes, missing stages, malformed PCM/digests, or text-bearing report
fields reject the cohort. Ten distinct successful cases are the hard minimum;
smaller clean corpora remain on hold.

Run its dependency-injected fake suite with:

```sh
PYTHONPATH=voice python -m unittest voice.test_shadow
```

Run the concrete ten-case cohort with the Faster Whisper environment:

```sh
cd /workspace/zer0/products/pm
./voice/shadow-real --report bench/voice-shadow-latest.json
```

`shadow_real.py` creates its spoken corpus from twelve fixed synthetic turns in
memory, decodes it through the candidate Faster Whisper model and `TurnOwner`,
starts one fresh ephemeral Codex thread per case on a private stdio app-server,
synthesizes each response through Kokoro, then hashes and discards the returned
PCM. It has no microphone, speaker, production session/thread, shared-daemon,
or audio-output option. `--report` atomically persists only the validated,
transcript-free JSON report; it never stores audio. A cohort smaller than ten
is rejected before adapters run. The JSON report contains only stage timings,
counts, safe adapter labels, and PCM sizes/hashes. It is directly consumable by
`release.verdict_from_canary`; binding it to a bundle digest and changing the
production pointer remain separate explicit release operations.

## Performance ledger

Pass `--metrics bench/voice-history.jsonl` to append one privacy-safe latency
record per turn. Records contain routing disposition and stage timings, never
transcript text or audio. Verify the current ledger with:

```sh
python voice/verify_metrics.py
```

The verifier gates only the current `codex-continuous-pcm-v5` pipeline and
the current runtime `run_id` while
retaining older rows as immutable historical cohorts. A rewrite therefore
starts red with zero current samples; it cannot inherit a passing result from an
obsolete pipeline or become green by deleting a regression. Promotion requires
ten recent completed samples, and every stage's p95—not merely its median—must
remain within budget.

The initial p95 budgets are 500 ms for ASR, 500 ms for first TTS synthesis,
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
