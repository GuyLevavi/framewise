"""Postgres: tables, engine, and the Postgres implementation of JobRepository."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from shared.detection import Box, Detection
from shared.jobs import FrameDetections, Job, JobStatus, NotFoundError, Video

DATABASE_URL_ENV = "DATABASE_URL"


class Base(DeclarativeBase):
    pass


class VideoRow(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"))
    detector: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)


class DetectionRow(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"))
    frame_index: Mapped[int] = mapped_column(Integer)
    frame_time: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    __table_args__ = (
        Index("ix_detections_video_time", "video_id", "frame_time"),
        Index("ix_detections_job", "job_id"),
    )


def make_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(f"set {DATABASE_URL_ENV}, e.g. postgresql+psycopg://user:pw@host/db")
    return create_engine(url, pool_pre_ping=True)


def _job(row: JobRow, video_path: str) -> Job:
    return Job(
        id=row.id,
        video_id=row.video_id,
        video_path=video_path,
        detector=row.detector,
        status=JobStatus(row.status),
        error=row.error,
        attempts=row.attempts,
        created_at=row.created_at,
    )


class PostgresJobRepository:
    """One short transaction per call; safe to use from several worker processes."""

    def __init__(self, engine: Engine) -> None:
        self._session = sessionmaker(engine, expire_on_commit=False)

    def add_video(self, video: Video) -> None:
        with self._session.begin() as s:
            s.add(
                VideoRow(
                    id=video.id,
                    filename=video.filename,
                    path=video.path,
                    size_bytes=video.size_bytes,
                    created_at=video.created_at,
                )
            )

    def get_video(self, video_id: str) -> Video:
        with self._session() as s:
            row = s.get(VideoRow, video_id)
            if row is None:
                raise NotFoundError(video_id)
            return Video(row.filename, row.path, row.size_bytes, row.id, row.created_at)

    def add(self, job: Job) -> None:
        with self._session.begin() as s:
            s.add(
                JobRow(
                    id=job.id,
                    video_id=job.video_id,
                    detector=job.detector,
                    status=job.status.value,
                    error=job.error,
                    attempts=job.attempts,
                    created_at=job.created_at,
                )
            )

    def get(self, job_id: str) -> Job:
        with self._session() as s:
            found = s.execute(
                select(JobRow, VideoRow.path).join(VideoRow).where(JobRow.id == job_id)
            ).first()
            if found is None:
                raise NotFoundError(job_id)
            return _job(found[0], found[1])

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._session.begin() as s:
            s.execute(update(JobRow).where(JobRow.id == job_id).values(status=status, error=error))

    def claim_next(self) -> Job | None:
        with self._session.begin() as s:
            row = s.scalars(
                select(JobRow)
                .where(JobRow.status == JobStatus.PENDING)
                .order_by(JobRow.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)  # other workers skip this row instead of waiting
            ).first()
            if row is None:
                return None
            row.status = JobStatus.RUNNING
            row.attempts += 1
            row.started_at = datetime.now(UTC)
            video = s.get_one(VideoRow, row.video_id)
            return _job(row, video.path)

    def reclaim_stale(self, older_than_s: float, max_attempts: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_s)
        stale = (JobRow.status == JobStatus.RUNNING) & (JobRow.started_at < cutoff)
        with self._session.begin() as s:
            failed = s.execute(
                update(JobRow)
                .where(stale & (JobRow.attempts >= max_attempts))
                .values(status=JobStatus.FAILED, error="worker stopped responding")
                .returning(JobRow.id)
            ).all()
            retried = s.execute(
                update(JobRow)
                .where(stale)
                .values(status=JobStatus.PENDING, started_at=None)
                .returning(JobRow.id)
            ).all()
        return len(failed) + len(retried)

    def save_detections(self, job_id: str, frames: list[FrameDetections]) -> None:
        rows = [f for f in frames if f.detections]
        if not rows:
            return
        with self._session.begin() as s:
            video_id = s.get_one(JobRow, job_id).video_id
            s.add_all(
                DetectionRow(
                    job_id=job_id,
                    video_id=video_id,
                    frame_index=f.frame_index,
                    frame_time=f.time_s,
                    label=d.label,
                    score=d.score,
                    x1=d.box.x1,
                    y1=d.box.y1,
                    x2=d.box.x2,
                    y2=d.box.y2,
                )
                for f in rows
                for d in f.detections
            )

    def list_detections(self, job_id: str) -> list[FrameDetections]:
        with self._session() as s:
            rows = s.scalars(
                select(DetectionRow)
                .where(DetectionRow.job_id == job_id)
                .order_by(DetectionRow.frame_index, DetectionRow.id)
            ).all()
        frames: dict[int, FrameDetections] = {}
        for r in rows:
            frame = frames.setdefault(
                r.frame_index, FrameDetections(r.frame_index, r.frame_time, [])
            )
            frame.detections.append(Detection(r.label, r.score, Box(r.x1, r.y1, r.x2, r.y2)))
        return list(frames.values())
