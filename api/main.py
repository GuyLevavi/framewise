"""HTTP API: upload videos, submit analysis jobs, read results."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from api.settings import ApiSettings, load_settings
from shared.auth import require_api_key, warn_if_open
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


class LabelSummaryOut(BaseModel):
    label: str
    detections: int
    first_seen_s: float
    last_seen_s: float


class Services:
    """Everything the endpoints need, built once at startup."""

    def __init__(self, settings: ApiSettings, repository: JobRepository) -> None:
        self.settings = settings
        self.detectors = load_yaml(settings.detectors_config, DetectorsFile)
        self.repository = repository
        settings.upload_dir.mkdir(parents=True, exist_ok=True)


def save_upload(file: UploadFile, target: Path, max_bytes: int) -> None:
    """Stream to disk in chunks, never the whole file in memory; stop as soon as it is too big."""
    written = 0
    with target.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                target.unlink()
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"upload larger than {max_bytes} bytes",
                )
            out.write(chunk)


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
        warn_if_open()
        yield

    # Every endpoint except /health (which OpenShift probes call) needs the API key.
    app = FastAPI(title="Framewise", lifespan=lifespan)
    protected = APIRouter(dependencies=[Depends(require_api_key)])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @protected.post("/videos", status_code=status.HTTP_201_CREATED)
    def upload_video(file: UploadFile, services: ServicesDep) -> VideoOut:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in services.settings.allowed_extensions:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"{suffix or 'no extension'} not allowed; use one of "
                f"{services.settings.allowed_extensions}",
            )
        video_id = uuid.uuid4().hex
        target = services.settings.upload_dir / f"{video_id}{suffix}"
        save_upload(file, target, services.settings.max_upload_mb * 1024 * 1024)
        video = Video(
            id=video_id,
            filename=file.filename or "",
            path=str(target.resolve()),
            size_bytes=target.stat().st_size,
        )
        services.repository.add_video(video)
        return VideoOut(video_id=video.id, filename=video.filename, size_bytes=video.size_bytes)

    @protected.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(body: JobIn, services: ServicesDep) -> JobOut:
        if body.detector not in services.detectors.detectors:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown detector {body.detector}"
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

    @protected.get("/videos/{video_id}/summary")
    def video_summary(video_id: str, services: ServicesDep) -> list[LabelSummaryOut]:
        try:
            services.repository.get_video(video_id)
        except NotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown video {video_id}") from None
        return [
            LabelSummaryOut(
                label=s.label,
                detections=s.detections,
                first_seen_s=s.first_seen_s,
                last_seen_s=s.last_seen_s,
            )
            for s in services.repository.summarize_video(video_id)
        ]

    @protected.get("/jobs/{job_id}")
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

    app.include_router(protected)
    return app


app = create_app()
