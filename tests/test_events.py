"""Composite events in SQL, against a real Postgres with hand-placed boxes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from shared.db import PostgresJobRepository
from shared.detection import Box, Detection
from shared.events import Rule, episode_key, find_episodes
from shared.jobs import FrameDetections, Job, JobStatus, Video

PERSON = Detection("person", 0.9, Box(0, 0, 10, 10))
CAR_TOUCHING = Detection("car", 0.9, Box(5, 0, 25, 10))  # overlaps half the person box
CAR_FAR = Detection("car", 0.9, Box(100, 100, 120, 120))


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    engine = create_engine(clean_db)
    yield engine
    engine.dispose()


def done_job(engine: Engine, frames: list[FrameDetections]) -> Job:
    repo = PostgresJobRepository(engine)
    video = Video(filename="v.avi", path="/v.avi", size_bytes=1)
    repo.add_video(video)
    job = Job(video_id=video.id, video_path=video.path, detector="x")
    repo.add(job)
    repo.save_detections(job.id, frames)
    repo.set_status(job.id, JobStatus.DONE)
    return job


def scene() -> list[FrameDetections]:
    """Overlap on 0.0-0.4 s and 1.0-1.2 s; at 0.6 s person and car are far apart."""
    together = [
        FrameDetections(i, i / 10, [PERSON, CAR_TOUCHING]) for i in (0, 1, 2, 3, 4, 10, 11, 12)
    ]
    apart = [FrameDetections(6, 0.6, [PERSON, CAR_FAR])]
    return together + apart


def spans(engine: Engine, job: Job, rule: Rule) -> list[tuple[float, float, int]]:
    return [(e.start_s, e.end_s, e.frames) for e in find_episodes(engine, job.video_id, "r", rule)]


def test_overlapping_frames_become_episodes_and_far_apart_boxes_do_not(engine: Engine) -> None:
    job = done_job(engine, scene())
    assert spans(engine, job, Rule(a="person", b=["car"], max_gap_s=0.15)) == [
        (0.0, 0.4, 5),
        (1.0, 1.2, 3),
    ]


def test_larger_gap_merges_episodes(engine: Engine) -> None:
    job = done_job(engine, scene())
    assert spans(engine, job, Rule(a="person", b=["car"], max_gap_s=0.7)) == [(0.0, 1.2, 8)]


def test_minimum_duration_drops_short_episodes(engine: Engine) -> None:
    job = done_job(engine, scene())
    rule = Rule(a="person", b=["car"], max_gap_s=0.15, min_duration_s=0.3)
    assert spans(engine, job, rule) == [(0.0, 0.4, 5)]


def test_overlap_threshold_is_relative_to_the_smaller_box(engine: Engine) -> None:
    # The overlap is 50 px: half the person box (100 px), a quarter of the car box (200 px).
    job = done_job(engine, scene())
    assert spans(engine, job, Rule(a="person", b=["car"], min_overlap=0.6)) == []  # only 50%
    found = spans(engine, job, Rule(a="person", b=["car"], min_overlap=0.4, max_gap_s=0.15))
    assert found == [(0.0, 0.4, 5), (1.0, 1.2, 3)], "0.4 passes for the smaller box, not the larger"


def test_keys_are_stable_across_reruns(engine: Engine) -> None:
    job = done_job(engine, scene())
    rule = Rule(a="person", b=["car"], max_gap_s=0.15)
    first = [e.key for e in find_episodes(engine, job.video_id, "r", rule)]
    again = [e.key for e in find_episodes(engine, job.video_id, "r", rule)]
    assert first == again and len(set(first)) == 2
    assert first[0] == episode_key(job.video_id, "r", 0.0)


def test_events_api_returns_episodes_with_keys(engine: Engine) -> None:
    from fastapi.testclient import TestClient

    from events.main import create_app
    from shared.events import RulesFile

    job = done_job(engine, scene())
    rules = RulesFile(rules={"person-car": Rule(a="person", b=["car"], max_gap_s=0.15)})
    with TestClient(create_app(rules, engine)) as client:
        body = client.get(f"/videos/{job.video_id}/events").json()
        assert [(e["start_s"], e["end_s"]) for e in body] == [(0.0, 0.4), (1.0, 1.2)]
        assert client.get("/videos/nope/events").status_code == 404
        assert client.get(f"/videos/{job.video_id}/events?rule=nope").status_code == 404
