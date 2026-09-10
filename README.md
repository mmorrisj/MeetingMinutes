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
                                                                         └─► FastAPI ──► Postgres
                                                                             (optional)  + pgvector
```

* `audio/devices.py`  finds the *loopback* input for your speakers on each OS.
* `audio/capture.py`  streams mono 16 kHz float32 blocks from it.
* `audio/chunker.py`  batches blocks into fixed windows and skips silent ones.
* `transcribe/`        turns a chunk into timestamped `Segment`s (Whisper today; pluggable).
* `pipeline.py`        wires the above with a bounded queue so the device never drops frames.
* `sinks.py`           writes segments to the console, an append-only `.jsonl` file, and
                       optionally the storage API.
* `api/`               FastAPI service: create a meeting, append segments, end it, list/read,
                       semantic `/search`.
* `db/`                SQLAlchemy 2.0 models (`Meeting`, `Segment`) with a pgvector column.
* `embeddings/`        `Embedder` protocol and a local ONNX backend (fastembed).
* `alembic/`           schema migrations.

## Platform notes (read this first)

| OS | Loopback mechanism | Setup |
|----|--------------------|-------|
| Windows | WASAPI loopback | Nothing to install. Every output device is capturable. |
| Linux | PulseAudio / PipeWire `*.monitor` source | Nothing to install if `pactl list sources short` shows a monitor. |
| macOS | **None built in.** | Install [BlackHole](https://existential.audio/blackhole/), create a *Multi-Output Device* in Audio MIDI Setup with BlackHole + your speakers, select it as system output, then `--device BlackHole`. |

## Install

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e ".[whisper,server,embed,dev]"
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

## Storage service (FastAPI + Postgres)

The recorder works on its own, but to keep meetings queryable across sessions run the API:

```bash
docker compose up -d db            # Postgres 16 with pgvector on localhost:5432
alembic upgrade head               # create the schema
meetingminutes serve               # http://127.0.0.1:8000/docs
meetingminutes record --api-url http://127.0.0.1:8000 --title "Weekly sync"
```

`record` creates a meeting, posts each transcribed chunk's segments as they arrive, and marks
the meeting ended on Ctrl-C. The JSONL file is still written, so a dead API never loses text.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/meetings` | Start a meeting (`title`, `source_device`, `started_at` optional). |
| `POST` | `/meetings/{id}/segments` | Append up to 1000 segments. `409` once the meeting has ended. |
| `POST` | `/meetings/{id}/end` | Mark ended. Idempotent. |
| `GET`  | `/meetings` | Newest first, `limit`/`offset`. |
| `GET`  | `/meetings/{id}` | Meeting with its segments in time order. |
| `GET`  | `/search?q=...` | Semantic search over all segments (`limit`, optional `meeting_id`). |
| `POST` | `/embeddings/backfill` | Embed segments that have no vector or one from another model. |
| `GET`  | `/health` | Checks the database connection. |

Connection settings: `MM_DATABASE_URL` (default matches `docker-compose.yml`), `MM_DB_ECHO`.

### Semantic search

Segments are embedded on ingest with a local ONNX model
([`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), 384-d, ~70 MB,
downloaded on first start) and stored in `segments.embedding`, a `vector(384)` column with an
HNSW index on cosine distance. `/search` embeds the query the same way and returns the closest
segments with their meeting, timestamps and a cosine-similarity `score`:

```bash
curl 'http://127.0.0.1:8000/search?q=when+is+the+launch&limit=5'
```

* Without the `embed` extra the API still stores segments; `/search` returns `503`. Install it
  later and call `POST /embeddings/backfill` to embed the backlog.
* `MM_EMBEDDINGS_ENABLED=false` skips embedding entirely (e.g. an offline machine).
* Switching to another 384-d model: set `MM_EMBEDDING_MODEL`, restart, run the backfill (it
  re-embeds rows tagged with the old `embedding_model`). A model with a different width needs a
  migration that alters the column, then a backfill.

## Development

```bash
pytest          # unit tests need no audio hardware; API tests use an embedded Postgres (pgserver)
ruff check .
```

Set `MM_TEST_DATABASE_URL` to run the API tests against your own Postgres instead of the
embedded one. Migrations are checked for drift against the models in `tests/test_migrations.py`.

## Roadmap

1. **Hybrid search** – combine the vector ranking with Postgres full-text search for exact
   names and numbers, which pure embeddings handle poorly.
2. **VAD-aligned chunking** – cut on silence instead of fixed 10 s windows so words are
   never split at a boundary.
3. **Speaker diarization** – who said what (pyannote or similar).
4. **Summaries and action items** – LLM pass over the transcript at the end of a meeting.
5. **Microphone mix-in** – optionally capture your own mic too so both sides are recorded.

## A note on consent

Recording a meeting is regulated in many jurisdictions (several US states and most of the EU
require all-party consent). Tell participants they are being transcribed.
