"""Runtime configuration, loadable from environment variables or a .env file.

Every setting is prefixed with ``MM_`` in the environment, e.g. ``MM_CHUNK_SECONDS=15``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MM_", env_file=".env", extra="ignore")

    # --- audio capture ---
    sample_rate: int = Field(
        16_000, description="Capture rate in Hz. 16 kHz is what Whisper wants."
    )
    channels: int = Field(1, description="Channels to capture. Speech models expect mono.")
    block_seconds: float = Field(0.5, description="Size of each read from the audio device.")
    device: str | None = Field(
        None,
        description="Substring of the loopback device name to capture from. "
        "Defaults to the monitor of the default speaker.",
    )

    # --- chunking ---
    chunk_seconds: float = Field(10.0, description="Audio window handed to the transcriber.")
    silence_rms: float = Field(
        0.005,
        description="Chunks whose RMS is below this are skipped as silence (float32 PCM scale).",
    )

    # --- transcription ---
    whisper_model: str = Field("base.en", description="faster-whisper model name or path.")
    whisper_device: str = Field("auto", description="'cpu', 'cuda', or 'auto'.")
    whisper_compute_type: str = Field("int8", description="CTranslate2 compute type.")
    language: str | None = Field(None, description="Force a language code, or None to detect.")

    # --- output ---
    output_dir: Path = Field(Path("transcripts"), description="Where transcript files are written.")
    api_url: str | None = Field(
        None,
        description="Base URL of the MeetingMinutes API. When set, `record` also streams segments "
        "there, e.g. http://localhost:8000.",
    )

    # --- storage service ---
    database_url: str = Field(
        "postgresql+asyncpg://meetingminutes:meetingminutes@localhost:5432/meetingminutes",
        description="SQLAlchemy async URL for Postgres (matches docker-compose.yml by default).",
    )
    db_echo: bool = Field(False, description="Log every SQL statement.")


def get_settings() -> Settings:
    return Settings()
