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
) -> None:
    """Capture system audio and stream a transcript to the console and a JSONL file."""
    from meetingminutes.audio.capture import SoundcardLoopbackSource
    from meetingminutes.audio.chunker import Chunker
    from meetingminutes.audio.devices import find_loopback_device
    from meetingminutes.pipeline import Pipeline
    from meetingminutes.sinks import ConsoleSink, JsonlSink, MultiSink, session_transcript_path
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
    pipeline = Pipeline(source, chunker, transcriber, MultiSink(ConsoleSink(), JsonlSink(out_path)))

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


if __name__ == "__main__":
    app()
