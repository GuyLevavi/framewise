"""The worker: claim a job, run it, repeat. Run several copies to process jobs in parallel."""

from __future__ import annotations

import logging
import os
import signal
import threading
from pathlib import Path
from types import FrameType

from pydantic import BaseModel, ConfigDict, Field

from shared.config import DetectorsFile, detector_factory, load_yaml
from shared.db import PostgresJobRepository, make_engine
from shared.jobs import JobRepository
from shared.logs import configure_json_logging
from shared.pipeline import PipelineRunner, VideoFileSource

log = logging.getLogger("framewise.worker")

CONFIG_ENV = "FRAMEWISE_WORKER_CONFIG"
DEFAULT_CONFIG = Path("worker/configs/worker.yaml")


class WorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detectors_config: Path
    poll_interval_s: float = Field(default=1.0, gt=0)
    stale_after_s: float = Field(default=600, gt=0)
    max_attempts: int = Field(default=3, ge=1)


def run_forever(
    repository: JobRepository,
    runner: PipelineRunner,
    settings: WorkerSettings,
    stop: threading.Event,
) -> None:
    log.info("worker started")
    while not stop.is_set():
        if reclaimed := repository.reclaim_stale(settings.stale_after_s, settings.max_attempts):
            log.warning("reclaimed %d stale jobs", reclaimed)
        job = repository.claim_next()
        if job is None:
            stop.wait(settings.poll_interval_s)
            continue
        log.info("job claimed", extra={"job_id": job.id, "detector": job.detector})
        runner.run(job)
        finished = repository.get(job.id)
        log.info("job %s", finished.status.lower(), extra={"job_id": job.id})
    log.info("worker stopped")


def main() -> None:
    configure_json_logging()
    settings = load_yaml(Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)), WorkerSettings)
    detectors = load_yaml(settings.detectors_config, DetectorsFile)
    repository = PostgresJobRepository(make_engine())
    runner = PipelineRunner(
        sources=lambda job: VideoFileSource(Path(job.video_path), detectors.sample_fps),
        detectors=detector_factory(detectors),
        repository=repository,
    )
    stop = threading.Event()

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("signal %d: finishing current job, then stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)  # what OpenShift sends before killing a pod
    signal.signal(signal.SIGINT, on_signal)
    run_forever(repository, runner, settings, stop)


if __name__ == "__main__":
    main()
