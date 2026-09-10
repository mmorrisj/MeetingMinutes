"""Command-line entry point: ``meetingminutes devices`` and ``meetingminutes record``."""

from __future__ import annotations

import logging
import signal
from pathlib import Path

import typer

from meetingminutes.config import get_settings

app = typer.Typer(
    help="Transcribe whatever is playing through your speakers.", no_args_is_help=True
)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def devices(
    all_devices: bool = typer.Option(False, "--all", help="Include plain microphones too."),
) -> None:
    """List loopback (system-audio) capture devices."""
    from meetingminutes.audio.devices import list_loopback_devices

    try:
        found = list_loopback_devices(include_all=all_devices)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if not found:
        typer.echo("No devices found.")
        raise typer.Exit(code=1)
    for dev in found:
        tag = "loopback" if dev.is_loopback else "mic     "
        typer.echo(f"{tag}  {dev.channels}ch  {dev.name}")


@app.command()
def record(
    device: str | None = typer.Option(None, help="Substring of the device name to capture."),
    output: Path | None = typer.Option(None, help="Transcript .jsonl path (default: timestamped)."),
    model: str | None = typer.Option(
        None, help="faster-whisper model, e.g. base.en, small, medium."
    ),
    chunk_seconds: float | None = typer.Option(None, help="Seconds of audio per transcription."),
    api_url: str | None = typer.Option(
        None, help="Also stream segments to the MeetingMinutes API at this base URL."
    ),
    title: str | None = typer.Option(None, help="Meeting title stored with the API record."),
) -> None:
    """Capture system audio and stream a transcript to the console and a JSONL file."""
    from meetingminutes.audio.capture import SoundcardLoopbackSource
    from meetingminutes.audio.chunker import Chunker
    from meetingminutes.audio.devices import find_loopback_device
    from meetingminutes.pipeline import Pipeline
    from meetingminutes.sinks import (
        ApiSink,
        ConsoleSink,
        JsonlSink,
        MultiSink,
        session_transcript_path,
    )
    from meetingminutes.transcribe.whisper import FasterWhisperTranscriber

    settings = get_settings()
    device_name = device or settings.device
    try:
        dev = find_loopback_device(device_name)
    except (LookupError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    out_path = output or session_transcript_path(settings.output_dir)
    typer.echo(f"Capturing from: {dev.name}")
    typer.echo(f"Transcript:     {out_path}")
    typer.echo("Loading model... (first run downloads it)")

    transcriber = FasterWhisperTranscriber(
        model=model or settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.language,
    )
    source = SoundcardLoopbackSource(
        dev, sample_rate=settings.sample_rate, block_seconds=settings.block_seconds
    )
    chunker = Chunker(
        settings.sample_rate,
        chunk_seconds=chunk_seconds or settings.chunk_seconds,
        silence_rms=settings.silence_rms,
    )
    sinks: list = [ConsoleSink(), JsonlSink(out_path)]
    api_base = api_url or settings.api_url
    if api_base:
        try:
            api_sink = ApiSink(api_base, title=title, source_device=dev.name)
        except Exception as exc:  # noqa: BLE001 - connection refused, 5xx, bad URL...
            typer.secho(f"Could not reach the API at {api_base}: {exc}", fg="red", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"API meeting:    {api_base}/meetings/{api_sink.meeting_id}")
        sinks.append(api_sink)
    pipeline = Pipeline(source, chunker, transcriber, MultiSink(*sinks))

    def _handle_stop(_sig, _frame) -> None:
        typer.echo("\nStopping...", err=True)
        pipeline.stop()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    typer.echo("Listening. Press Ctrl-C to stop.\n")
    pipeline.run()
    typer.echo(
        f"\nDone: {pipeline.chunks_transcribed} chunks, {pipeline.segments_written} segments -> "
        f"{out_path}"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (development)."),
) -> None:
    """Run the storage API (FastAPI + Postgres). Run `alembic upgrade head` first."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        typer.secho("Install the server extra: pip install 'meetingminutes[server]'", fg="red")
        raise typer.Exit(code=1) from exc
    uvicorn.run("meetingminutes.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
