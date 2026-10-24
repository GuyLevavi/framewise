"""HTTP API: upload videos, submit analysis jobs, read results."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from api.settings import ApiSettings, load_settings
from shared.config import DetectorsFile, load_yaml
from shared.db import PostgresJobRepository, make_engine
from shared.jobs import Job, JobRepository, JobStatus, NotFoundError, Video
from shared.pipeline import group_segments


class VideoOut(BaseModel):
    video_id: str
    filename: str
    size_bytes: int


class JobIn(BaseModel):
    video_id: str
    detector: str = "motion-sensitive"


class SegmentOut(BaseModel):
    start_s: float
    end_s: float


class JobOut(BaseModel):
    id: str
    status: JobStatus
    error: str | None = None
    segments: list[SegmentOut] = []


class Services:
    """Everything the endpoints need, built once at startup."""

    def __init__(self, settings: ApiSettings, repository: JobRepository) -> None:
        self.settings = settings
        self.detectors = load_yaml(settings.detectors_config, DetectorsFile)
        self.repository = repository
        settings.upload_dir.mkdir(parents=True, exist_ok=True)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def create_app(
    settings: ApiSettings | None = None, repository: JobRepository | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = Services(
            settings or load_settings(), repository or PostgresJobRepository(make_engine())
        )
        yield

    app = FastAPI(title="Framewise", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/videos", status_code=status.HTTP_201_CREATED)
    def upload_video(file: UploadFile, services: ServicesDep) -> VideoOut:
        video_id = uuid.uuid4().hex
        suffix = Path(file.filename or "video").suffix or ".bin"
        target = services.settings.upload_dir / f"{video_id}{suffix}"
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)  # streams to disk; never the whole file in memory
        video = Video(
            id=video_id,
            filename=file.filename or "",
            path=str(target.resolve()),
            size_bytes=target.stat().st_size,
        )
        services.repository.add_video(video)
        return VideoOut(video_id=video.id, filename=video.filename, size_bytes=video.size_bytes)

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(body: JobIn, services: ServicesDep) -> JobOut:
        if body.detector not in services.detectors.detectors:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown detector {body.detector}"
            )
        try:
            video = services.repository.get_video(body.video_id)
        except NotFoundError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"unknown video {body.video_id}"
            ) from None
        job = Job(video_id=video.id, video_path=video.path, detector=body.detector)
        services.repository.add(job)  # a worker picks it up; the API never runs long work
        return JobOut(id=job.id, status=job.status)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, services: ServicesDep) -> JobOut:
        try:
            job = services.repository.get(job_id)
        except NotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown job {job_id}") from None
        segments: list[SegmentOut] = []
        if job.status is JobStatus.DONE:
            frames = services.repository.list_detections(job_id)
            times = [f.time_s for f in frames if f.detections]
            segments = [
                SegmentOut(start_s=s.start_s, end_s=s.end_s)
                for s in group_segments(times, services.detectors.max_gap_s)
            ]
        return JobOut(id=job.id, status=job.status, error=job.error, segments=segments)

    return app


app = create_app()
