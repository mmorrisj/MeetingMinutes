from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from meetingminutes.api.app import create_app
from meetingminutes.sinks import ApiSink
from meetingminutes.transcribe.base import Segment


@pytest.fixture
def sync_client(settings):
    # TestClient is a sync httpx.Client over the ASGI app and runs the lifespan on enter,
    # which is what ApiSink (a sync sink) needs.
    with TestClient(create_app(settings), base_url="http://test") as c:
        yield c


def test_sink_creates_meeting_streams_segments_and_ends(sync_client, settings):
    sink = ApiSink("http://test", title="Standup", source_device="BlackHole", client=sync_client)
    sink.write([Segment("one", 0.0, 1.0, 0.8), Segment("two", 1.0, 2.0)])
    sink.write([])  # empty batches are not posted
    sink.write([Segment("three", 2.0, 3.0)])
    sink.close()  # ends the meeting and closes the client

    with TestClient(create_app(settings)) as c:
        body = c.get(f"/meetings/{sink.meeting_id}").json()
    assert body["title"] == "Standup"
    assert body["source_device"] == "BlackHole"
    assert body["ended_at"] is not None
    assert [s["text"] for s in body["segments"]] == ["one", "two", "three"]


def test_sink_retries_then_gives_up_without_raising(monkeypatch, caplog):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path == "/meetings":
            return httpx.Response(201, json={"id": "abc"})
        return httpx.Response(503)

    monkeypatch.setattr("meetingminutes.sinks.time.sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    sink = ApiSink("http://x", client=client, retries=3)

    sink.write([Segment("hi", 0, 1)])  # must not raise
    assert calls["n"] == 1 + 3
    assert "giving up" in caplog.text
