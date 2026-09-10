# MeetingMinutes

Takes notes during meetings by transcribing **whatever is playing through your speakers**.
Because it listens to the system's audio output rather than to a specific app, it works the
same with Zoom, Teams, Meet, Webex, a browser tab, or a recording you are playing back.

Audio never leaves your machine: transcription runs locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## How it works

```
loopback device ──► 0.5 s blocks ──► 10 s chunks ──► faster-whisper ──► console + JSONL
 (OS specific)      (capture thread)   (silence dropped)   (main thread)     transcripts/
```

* `audio/devices.py`  finds the *loopback* input for your speakers on each OS.
* `audio/capture.py`  streams mono 16 kHz float32 blocks from it.
* `audio/chunker.py`  batches blocks into fixed windows and skips silent ones.
* `transcribe/`        turns a chunk into timestamped `Segment`s (Whisper today; pluggable).
* `pipeline.py`        wires the above with a bounded queue so the device never drops frames.
* `sinks.py`           writes segments to the console and an append-only `.jsonl` file.

## Platform notes (read this first)

| OS | Loopback mechanism | Setup |
|----|--------------------|-------|
| Windows | WASAPI loopback | Nothing to install. Every output device is capturable. |
| Linux | PulseAudio / PipeWire `*.monitor` source | Nothing to install if `pactl list sources short` shows a monitor. |
| macOS | **None built in.** | Install [BlackHole](https://existential.audio/blackhole/), create a *Multi-Output Device* in Audio MIDI Setup with BlackHole + your speakers, select it as system output, then `--device BlackHole`. |

## Install

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e ".[whisper,dev]"
```

## Use

```bash
meetingminutes devices            # what can I capture from?
meetingminutes record             # picks the loopback for your default speaker
meetingminutes record --device BlackHole --model small.en --chunk-seconds 15
```

Transcripts land in `transcripts/meeting-<UTC timestamp>.jsonl`, one segment per line:

```json
{"text": "Let's move the launch to Thursday.", "start_seconds": 812.4, "end_seconds": 815.1, "confidence": 0.91}
```

Settings can also come from the environment or a `.env` file, prefixed `MM_`
(`MM_WHISPER_MODEL=small.en`, `MM_CHUNK_SECONDS=15`, ...). See `config.py`.

## Development

```bash
pytest          # unit tests run with no audio hardware
ruff check .
```

## Roadmap

1. **Storage + search** – FastAPI service, Postgres + pgvector, SQLAlchemy models for
   meetings and segments, embeddings for semantic search over past meetings.
2. **VAD-aligned chunking** – cut on silence instead of fixed 10 s windows so words are
   never split at a boundary.
3. **Speaker diarization** – who said what (pyannote or similar).
4. **Summaries and action items** – LLM pass over the transcript at the end of a meeting.
5. **Microphone mix-in** – optionally capture your own mic too so both sides are recorded.

## A note on consent

Recording a meeting is regulated in many jurisdictions (several US states and most of the EU
require all-party consent). Tell participants they are being transcribed.
