"""Jobs and where they are stored."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from shared.detection import Detection


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Job:
    video_path: str
    detector: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class FrameDetections:
    """All detections for one sampled frame."""

    frame_index: int
    time_s: float
    detections: list[Detection]


class JobNotFoundError(KeyError):
    pass


class JobRepository(Protocol):
    def add(self, job: Job) -> None: ...
    def get(self, job_id: str) -> Job: ...
    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None: ...
    def claim_next(self) -> Job | None:
        """Atomically move the oldest PENDING job to RUNNING and return it."""
        ...

    def save_detections(self, job_id: str, frames: list[FrameDetections]) -> None: ...
    def list_detections(self, job_id: str) -> list[FrameDetections]: ...


class InMemoryJobRepository:
    """For tests and the CLI. Not shared between processes."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._detections: dict[str, list[FrameDetections]] = {}

    def add(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(job_id) from None

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        self._jobs[job_id] = replace(self.get(job_id), status=status, error=error)

    def claim_next(self) -> Job | None:
        pending = [j for j in self._jobs.values() if j.status is JobStatus.PENDING]
        if not pending:
            return None
        job = min(pending, key=lambda j: j.created_at)
        self.set_status(job.id, JobStatus.RUNNING)
        return self.get(job.id)

    def save_detections(self, job_id: str, frames: list[FrameDetections]) -> None:
        self._detections.setdefault(job_id, []).extend(frames)

    def list_detections(self, job_id: str) -> list[FrameDetections]:
        return list(self._detections.get(job_id, []))
