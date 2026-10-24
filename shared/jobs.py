"""Videos, jobs and their results, and the interface for storing them."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from shared.detection import Detection


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Video:
    filename: str
    path: str
    size_bytes: int
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class Job:
    video_id: str
    video_path: str
    detector: str
    id: str = field(default_factory=_new_id)
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    attempts: int = 0
    created_at: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class FrameDetections:
    """Detections for one sampled frame. Frames without detections are not stored."""

    frame_index: int
    time_s: float
    detections: list[Detection]


class NotFoundError(KeyError):
    pass


class JobRepository(Protocol):
    def add_video(self, video: Video) -> None: ...
    def get_video(self, video_id: str) -> Video: ...
    def add(self, job: Job) -> None: ...
    def get(self, job_id: str) -> Job: ...
    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None: ...
    def claim_next(self) -> Job | None:
        """Atomically move the oldest PENDING job to RUNNING and return it."""
        ...

    def reclaim_stale(self, older_than_s: float, max_attempts: int) -> int:
        """Put timed-out RUNNING jobs back to PENDING, or FAILED after max_attempts."""
        ...

    def save_detections(self, job_id: str, frames: list[FrameDetections]) -> None: ...
    def list_detections(self, job_id: str) -> list[FrameDetections]: ...


class InMemoryJobRepository:
    """For unit tests and the CLI. Not shared between processes."""

    def __init__(self) -> None:
        self._videos: dict[str, Video] = {}
        self._jobs: dict[str, Job] = {}
        self._started: dict[str, datetime] = {}
        self._detections: dict[str, list[FrameDetections]] = {}

    def add_video(self, video: Video) -> None:
        self._videos[video.id] = video

    def get_video(self, video_id: str) -> Video:
        try:
            return self._videos[video_id]
        except KeyError:
            raise NotFoundError(video_id) from None

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise NotFoundError(job_id) from None

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        self._jobs[job_id] = replace(self.get(job_id), status=status, error=error)

    def claim_next(self) -> Job | None:
        pending = [j for j in self._jobs.values() if j.status is JobStatus.PENDING]
        if not pending:
            return None
        job = min(pending, key=lambda j: j.created_at)
        claimed = replace(job, status=JobStatus.RUNNING, attempts=job.attempts + 1)
        self._jobs[job.id] = claimed
        self._started[job.id] = _now()
        return claimed

    def reclaim_stale(self, older_than_s: float, max_attempts: int) -> int:
        count = 0
        for job in list(self._jobs.values()):
            started = self._started.get(job.id)
            if job.status is not JobStatus.RUNNING or started is None:
                continue
            if (_now() - started).total_seconds() < older_than_s:
                continue
            if job.attempts >= max_attempts:
                self.set_status(job.id, JobStatus.FAILED, "worker stopped responding")
            else:
                self.set_status(job.id, JobStatus.PENDING)
            count += 1
        return count

    def save_detections(self, job_id: str, frames: list[FrameDetections]) -> None:
        self._detections.setdefault(job_id, []).extend(f for f in frames if f.detections)

    def list_detections(self, job_id: str) -> list[FrameDetections]:
        return sorted(self._detections.get(job_id, []), key=lambda f: f.frame_index)
