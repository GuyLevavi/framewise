"""PostgresJobRepository against a real Postgres (see checks/fixtures.py: database_url)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from shared.db import PostgresJobRepository
from shared.detection import Box, Detection
from shared.jobs import FrameDetections, Job, JobStatus, NotFoundError, Video
from shared.logs import JsonFormatter


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    engine = create_engine(clean_db)
    yield engine
    engine.dispose()


@pytest.fixture
def repo(engine: Engine) -> PostgresJobRepository:
    return PostgresJobRepository(engine)


def add_jobs(repo: PostgresJobRepository, n: int) -> list[Job]:
    video = Video(filename="v.avi", path="/data/v.avi", size_bytes=1)
    repo.add_video(video)
    jobs = [
        Job(video_id=video.id, video_path=video.path, detector="motion-sensitive") for _ in range(n)
    ]
    for job in jobs:
        repo.add(job)
    return jobs


def test_round_trip(repo: PostgresJobRepository) -> None:
    [job] = add_jobs(repo, 1)
    assert repo.get(job.id).status is JobStatus.PENDING
    assert repo.get(job.id).video_path == "/data/v.avi"
    frames = [
        FrameDetections(3, 0.3, [Detection("person", 0.9, Box(1, 2, 3, 4))]),
        FrameDetections(4, 0.4, []),
    ]
    repo.save_detections(job.id, frames)
    stored = repo.list_detections(job.id)
    assert [(f.frame_index, f.detections) for f in stored] == [(3, frames[0].detections)]
    with pytest.raises(NotFoundError):
        repo.get("missing")


def test_two_workers_never_claim_the_same_job(engine: Engine) -> None:
    setup = PostgresJobRepository(engine)
    jobs = add_jobs(setup, 40)
    claimed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        repo = PostgresJobRepository(engine)  # each "worker" uses its own connections
        while (job := repo.claim_next()) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(j.id for j in jobs)  # every job once, none twice


def test_stale_jobs_are_reclaimed_then_failed(repo: PostgresJobRepository, engine: Engine) -> None:
    [job] = add_jobs(repo, 1)

    def claim_and_die() -> None:
        assert repo.claim_next() is not None
        with engine.begin() as conn:  # pretend the worker died an hour ago
            conn.execute(text("UPDATE jobs SET started_at = now() - interval '1 hour'"))

    claim_and_die()
    assert repo.reclaim_stale(older_than_s=60, max_attempts=2) == 1
    assert repo.get(job.id).status is JobStatus.PENDING
    claim_and_die()
    assert repo.reclaim_stale(older_than_s=60, max_attempts=2) == 1
    failed = repo.get(job.id)
    assert failed.status is JobStatus.FAILED and failed.attempts == 2


def test_fresh_running_job_is_not_reclaimed(repo: PostgresJobRepository) -> None:
    add_jobs(repo, 1)
    assert repo.claim_next() is not None
    assert repo.reclaim_stale(older_than_s=60, max_attempts=3) == 0


def test_detections_by_video_and_time_use_the_index(
    repo: PostgresJobRepository, engine: Engine
) -> None:
    [job] = add_jobs(repo, 1)
    frames = [
        FrameDetections(i, i / 10, [Detection("person", 0.9, Box(0, 0, 1, 1))])
        for i in range(20_000)
    ]
    repo.save_detections(job.id, frames)
    with engine.begin() as conn:
        conn.execute(text("ANALYZE detections"))
        plan = "\n".join(
            conn.execute(
                text(
                    "EXPLAIN SELECT * FROM detections "
                    "WHERE video_id = :v AND frame_time BETWEEN 10 AND 12"
                ),
                {"v": job.video_id},
            ).scalars()
        )
    assert "ix_detections_video_time" in plan, plan


def test_json_log_line_carries_job_id() -> None:
    record = logging.LogRecord("framewise", logging.INFO, __file__, 1, "job done", None, None)
    record.job_id = "abc123"
    line = json.loads(JsonFormatter().format(record))
    assert line["job_id"] == "abc123" and line["event"] == "job done" and line["level"] == "INFO"


def test_summary_counts_labels_with_first_and_last_seen(repo: PostgresJobRepository) -> None:
    [job] = add_jobs(repo, 1)
    person, car = Detection("person", 0.9, Box(0, 0, 1, 1)), Detection("car", 0.8, Box(0, 0, 2, 2))
    repo.save_detections(
        job.id,
        [FrameDetections(1, 0.1, [person]), FrameDetections(5, 0.5, [person, car])],
    )
    summary = {s.label: s for s in repo.summarize_video(job.video_id)}
    assert (summary["person"].detections, summary["person"].first_seen_s) == (2, 0.1)
    assert (summary["car"].detections, summary["car"].last_seen_s) == (1, 0.5)
