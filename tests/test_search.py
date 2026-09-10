from __future__ import annotations

import pytest
from sqlalchemy import select

from meetingminutes.db.models import Segment


async def _meeting_with(client, title, texts):
    meeting = (await client.post("/meetings", json={"title": title})).json()
    segs = [
        {"text": t, "start_seconds": float(i), "end_seconds": float(i + 1)}
        for i, t in enumerate(texts)
    ]
    resp = await client.post(f"/meetings/{meeting['id']}/segments", json={"segments": segs})
    assert resp.status_code == 200, resp.text
    return meeting


async def test_ingest_embeds_segments(client, app, embedder):
    await _meeting_with(client, "m", ["hello world"])
    async with app.state.session_factory() as session:
        seg = await session.scalar(select(Segment))
    assert seg.embedding_model == embedder.model_name
    assert len(seg.embedding) == embedder.dim


async def test_search_ranks_closest_first_and_reports_meeting(client):
    planning = await _meeting_with(
        client,
        "Planning",
        ["the launch moved to thursday", "budget review is next quarter"],
    )
    await _meeting_with(client, "Lunch", ["pizza arrived late", "coffee machine is broken"])

    body = (await client.get("/search", params={"q": "launch thursday", "limit": 3})).json()
    assert body["query"] == "launch thursday"
    top = body["hits"][0]
    assert top["text"] == "the launch moved to thursday"
    assert top["meeting_id"] == planning["id"]
    assert top["meeting_title"] == "Planning"
    assert top["start_seconds"] == 0.0
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[1]


async def test_search_can_be_scoped_to_one_meeting(client):
    a = await _meeting_with(client, "A", ["coffee machine is broken"])
    b = await _meeting_with(client, "B", ["coffee machine is fixed"])

    body = (
        await client.get("/search", params={"q": "coffee machine", "meeting_id": b["id"]})
    ).json()
    assert [h["meeting_id"] for h in body["hits"]] == [b["id"]]
    assert a["id"] not in {h["meeting_id"] for h in body["hits"]}


async def test_search_validation(client):
    assert (await client.get("/search")).status_code == 422
    assert (await client.get("/search", params={"q": "", "limit": 1})).status_code == 422
    assert (await client.get("/search", params={"q": "x", "limit": 0})).status_code == 422


class TestWithoutEmbedder:
    @pytest.fixture
    def embedder(self):
        return None

    async def test_ingest_works_and_search_is_503(self, client, app):
        await _meeting_with(client, "m", ["stored without a vector"])
        async with app.state.session_factory() as session:
            seg = await session.scalar(select(Segment))
        assert seg.embedding is None and seg.embedding_model is None

        resp = await client.get("/search", params={"q": "vector"})
        assert resp.status_code == 503
        assert (await client.post("/embeddings/backfill")).status_code == 503


async def test_backfill_embeds_missing_and_stale_rows(client, app, embedder):
    from tests.conftest import HashEmbedder

    # Ingest without an embedder, then one with an older model name.
    app.state.embedder = None
    await _meeting_with(client, "old", ["no vector yet", "me neither"])

    stale = HashEmbedder()
    stale.model_name = "test-hash-v0"
    app.state.embedder = stale
    await _meeting_with(client, "stale", ["embedded by an old model"])

    app.state.embedder = embedder
    await _meeting_with(client, "fresh", ["already current"])

    body = (await client.post("/embeddings/backfill", params={"batch_size": 2})).json()
    assert body == {"embedded": 3, "remaining": False}

    async with app.state.session_factory() as session:
        models = set(await session.scalars(select(Segment.embedding_model)))
    assert models == {embedder.model_name}

    # Everything is now searchable.
    body = (await client.get("/search", params={"q": "no vector yet"})).json()
    assert body["hits"][0]["text"] == "no vector yet"

    # Idempotent.
    assert (await client.post("/embeddings/backfill")).json() == {"embedded": 0, "remaining": False}


async def test_backfill_respects_max_segments(client, app):
    app.state.embedder = None
    await _meeting_with(client, "m", ["a", "b", "c"])
    from tests.conftest import HashEmbedder

    app.state.embedder = HashEmbedder()
    body = (await client.post("/embeddings/backfill", params={"max_segments": 2})).json()
    assert body == {"embedded": 2, "remaining": True}
