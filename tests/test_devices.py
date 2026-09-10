from __future__ import annotations

from types import SimpleNamespace

import pytest

from meetingminutes.audio import devices as dev_mod


class FakeSoundcard:
    def __init__(self, mics, default_speaker="Speakers (Realtek)"):
        self._mics = mics
        self._default = default_speaker

    def all_microphones(self, include_loopback=False):
        return self._mics

    def default_speaker(self):
        return SimpleNamespace(name=self._default)


def _mic(name, isloopback=False, channels=2, id=None):
    return SimpleNamespace(name=name, isloopback=isloopback, channels=channels, id=id or name)


@pytest.fixture
def patch_sc(monkeypatch):
    def _install(mics, default_speaker="Speakers (Realtek)"):
        fake = FakeSoundcard(mics, default_speaker)
        monkeypatch.setattr(dev_mod, "_soundcard", lambda: fake)
        return fake

    return _install


def test_windows_prefers_loopback_of_default_speaker(patch_sc):
    patch_sc(
        [
            _mic("Headset Mic"),
            _mic("Monitor Speakers", isloopback=True),
            _mic("Speakers (Realtek)", isloopback=True),
        ]
    )
    assert dev_mod.find_loopback_device().name == "Speakers (Realtek)"


def test_linux_monitor_source_is_detected(patch_sc):
    patch_sc(
        [_mic("Built-in Audio Analog Stereo"), _mic("Monitor of Built-in Audio Analog Stereo")],
        default_speaker="",
    )
    found = dev_mod.list_loopback_devices()
    assert [d.name for d in found] == ["Monitor of Built-in Audio Analog Stereo"]
    assert found[0].is_loopback


def test_macos_blackhole_counts_as_loopback(patch_sc):
    patch_sc(
        [_mic("MacBook Pro Microphone"), _mic("BlackHole 2ch")],
        default_speaker="MacBook Pro Speakers",
    )
    assert dev_mod.find_loopback_device().name == "BlackHole 2ch"


def test_name_filter_matches_any_device_and_errors_clearly(patch_sc):
    patch_sc([_mic("USB Mic"), _mic("BlackHole 2ch")])
    assert dev_mod.find_loopback_device("usb").name == "USB Mic"
    with pytest.raises(LookupError, match="devices --all"):
        dev_mod.find_loopback_device("nope")


def test_helpful_error_when_nothing_looks_like_loopback(patch_sc, monkeypatch):
    patch_sc([_mic("MacBook Pro Microphone")])
    monkeypatch.setattr(dev_mod.sys, "platform", "darwin")
    with pytest.raises(LookupError, match="BlackHole"):
        dev_mod.find_loopback_device()
