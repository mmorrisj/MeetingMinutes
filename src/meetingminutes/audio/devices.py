"""Discovery of *loopback* audio devices, i.e. inputs that carry what the speakers are playing.

How that works differs per OS, and this module hides the differences:

* **Windows** - WASAPI exposes every render endpoint as a loopback capture endpoint.
  ``soundcard`` surfaces them via ``all_microphones(include_loopback=True)``.
* **Linux** - PulseAudio / PipeWire create a ``*.monitor`` source for every sink.
  They show up as ordinary microphones whose name ends in ``Monitor``/``.monitor``.
* **macOS** - CoreAudio has no loopback. You need a virtual device such as BlackHole
  (https://existential.audio/blackhole/) configured as a Multi-Output Device so audio still
  reaches your speakers. It then appears as a normal microphone named "BlackHole".

The ``soundcard`` import is deferred so the rest of the package (and the test-suite) works on
machines with no audio stack at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

_MAC_VIRTUAL_DEVICE_HINTS = ("blackhole", "loopback", "soundflower", "vb-cable")


@dataclass(frozen=True, slots=True)
class LoopbackDevice:
    id: str
    name: str
    channels: int
    is_loopback: bool


def _soundcard():
    try:
        import soundcard  # noqa: PLC0415  (deferred on purpose)
    except (ImportError, OSError) as exc:  # OSError: missing libpulse / CoreAudio bindings
        raise RuntimeError(
            "The 'soundcard' library could not be loaded. On Linux install PulseAudio/PipeWire "
            "client libraries (e.g. libpulse0); on macOS/Windows reinstall with "
            "`pip install soundcard`."
        ) from exc
    return soundcard


def _looks_like_loopback(name: str, native_flag: bool) -> bool:
    lowered = name.lower()
    if native_flag:
        return True
    if lowered.endswith(".monitor") or "monitor of" in lowered or lowered.endswith(" monitor"):
        return True
    return any(hint in lowered for hint in _MAC_VIRTUAL_DEVICE_HINTS)


def list_loopback_devices(*, include_all: bool = False) -> list[LoopbackDevice]:
    """Return capture devices, loopback/monitor ones first.

    With ``include_all=False`` (default) only devices that look like loopbacks are returned.
    """
    sc = _soundcard()
    mics = sc.all_microphones(include_loopback=True)
    devices: list[LoopbackDevice] = []
    for mic in mics:
        native_flag = bool(getattr(mic, "isloopback", False))
        is_loop = _looks_like_loopback(mic.name, native_flag)
        if is_loop or include_all:
            devices.append(
                LoopbackDevice(
                    id=str(mic.id), name=mic.name, channels=int(mic.channels), is_loopback=is_loop
                )
            )
    devices.sort(key=lambda d: (not d.is_loopback, d.name.lower()))
    return devices


def find_loopback_device(name_filter: str | None = None) -> LoopbackDevice:
    """Pick the loopback device to record from.

    Preference order:
      1. A device whose name contains ``name_filter`` (case-insensitive), if given.
      2. The loopback that corresponds to the current default speaker.
      3. The first loopback device found.
    """
    devices = list_loopback_devices(include_all=name_filter is not None)

    if name_filter:
        wanted = name_filter.lower()
        for dev in devices:
            if wanted in dev.name.lower() or wanted == dev.id.lower():
                return dev
        raise LookupError(
            f"No capture device matches {name_filter!r}. "
            "Run `meetingminutes devices --all` to see what is available."
        )

    loopbacks = [d for d in devices if d.is_loopback]
    if not loopbacks:
        raise LookupError(_no_loopback_help())

    try:
        default_speaker = _soundcard().default_speaker().name.lower()
    except Exception:  # noqa: BLE001 - any failure here just means "no preference"
        default_speaker = ""
    if default_speaker:
        for dev in loopbacks:
            if default_speaker in dev.name.lower():
                return dev
    return loopbacks[0]


def _no_loopback_help() -> str:
    if sys.platform == "darwin":
        return (
            "No loopback device found. macOS has no built-in system-audio capture; install "
            "BlackHole (https://existential.audio/blackhole/), create a Multi-Output Device in "
            "Audio MIDI Setup containing both BlackHole and your speakers, select it as system "
            "output, then run `meetingminutes record --device BlackHole`."
        )
    if sys.platform.startswith("linux"):
        return (
            "No monitor source found. Make sure PulseAudio or PipeWire is running "
            "(`pactl list sources short` should list a '*.monitor' entry)."
        )
    return (
        "No WASAPI loopback endpoint found. Make sure an output device is enabled in "
        "Windows sound settings."
    )
