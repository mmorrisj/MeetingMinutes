from __future__ import annotations

import uuid

SEGS = [
    {"text": "Welcome everyone.", "start_seconds": 0.0, "end_seconds": 1.4, "confidence": 0.9},
    {"text": "Let's start with the roadmap.", "start_seconds": 1.6, "end_seconds": 3.9},
]


async def _create(client, **body):
    resp = await client.post("/meetings", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_create_append_and_fetch(client):
    meeting = await _create(client, title="Weekly sync", source_device="Speakers")
    assert meeting["ended_at"] is None
    assert meeting["started_at"]

    resp = await client.post(f"/meetings/{meeting['id']}/segments", json={"segments": SEGS})
    assert resp.status_code == 200
    assert resp.json() == {"inserted": 2}

    resp = await client.get(f"/meetings/{meeting['id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Weekly sync"
    assert [s["text"] for s in body["segments"]] == [s["text"] for s in SEGS]
    assert body["segments"][0]["confidence"] == 0.9
    assert body["segments"][1]["confidence"] is None


async def test_segments_come_back_in_time_order(client):
    meeting = await _create(client)
    later = {"text": "later", "start_seconds": 30.0, "end_seconds": 31.0}
    earlier = {"text": "earlier", "start_seconds": 5.0, "end_seconds": 6.0}
    await client.post(f"/meetings/{meeting['id']}/segments", json={"segments": [later]})
    await client.post(f"/meetings/{meeting['id']}/segments", json={"segments": [earlier]})

    body = (await client.get(f"/meetings/{meeting['id']}")).json()
    assert [s["text"] for s in body["segments"]] == ["earlier", "later"]


async def test_list_is_newest_first_and_paginated(client):
    for i in range(3):
        await _create(client, title=f"m{i}", started_at=f"2026-01-0{i + 1}T10:00:00Z")

    body = (await client.get("/meetings", params={"limit": 2})).json()
    assert body["total"] == 3
    assert [m["title"] for m in body["items"]] == ["m2", "m1"]

    body = (await client.get("/meetings", params={"limit": 2, "offset": 2})).json()
    assert [m["title"] for m in body["items"]] == ["m0"]


async def test_end_is_idempotent_and_blocks_further_segments(client):
    meeting = await _create(client)
    first = (await client.post(f"/meetings/{meeting['id']}/end")).json()
    second = (await client.post(f"/meetings/{meeting['id']}/end")).json()
    assert first["ended_at"] is not None
    assert first["ended_at"] == second["ended_at"]

    resp = await client.post(f"/meetings/{meeting['id']}/segments", json={"segments": SEGS})
    assert resp.status_code == 409


async def test_unknown_meeting_is_404(client):
    missing = uuid.uuid4()
    assert (await client.get(f"/meetings/{missing}")).status_code == 404
    assert (await client.post(f"/meetings/{missing}/end")).status_code == 404
    resp = await client.post(f"/meetings/{missing}/segments", json={"segments": SEGS})
    assert resp.status_code == 404


async def test_segment_validation(client):
    meeting = await _create(client)
    url = f"/meetings/{meeting['id']}/segments"
    bad = [
        {"segments": []},
        {"segments": [{"text": "", "start_seconds": 0, "end_seconds": 1}]},
        {"segments": [{"text": "x", "start_seconds": 2, "end_seconds": 1}]},
        {"segments": [{"text": "x", "start_seconds": 0, "end_seconds": 1, "confidence": 1.5}]},
    ]
    for body in bad:
        assert (await client.post(url, json=body)).status_code == 422, body
