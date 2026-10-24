"""Running a job: frames in, detections out, status tracked."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from shared.detection import Detector
from shared.jobs import FrameDetections, Job, JobRepository, JobStatus
from shared.video import Frame, iter_frames, open_video

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    def frames(self) -> Iterator[tuple[int, float, Frame]]:
        """Yield (frame_index, time_s, BGR frame)."""
        ...


class VideoFileSource:
    """Samples a video file at roughly `sample_fps` frames per second."""

    def __init__(self, path: Path, sample_fps: float) -> None:
        self._path = path
        self._sample_fps = sample_fps

    def frames(self) -> Iterator[tuple[int, float, Frame]]:
        with open_video(self._path) as capture:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        every_n = max(1, round(fps / self._sample_fps))
        for index, frame in iter_frames(self._path, every_n):
            yield index, index / fps, frame


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float


def group_segments(times_s: list[float], max_gap_s: float) -> list[Segment]:
    """Merge sorted timestamps into segments; a gap larger than max_gap_s starts a new one."""
    segments: list[Segment] = []
    for t in sorted(times_s):
        if segments and t - segments[-1].end_s <= max_gap_s:
            segments[-1] = Segment(segments[-1].start_s, t)
        else:
            segments.append(Segment(t, t))
    return segments


DetectorFactory = Callable[[str], Detector]
SourceFactory = Callable[[Job], FrameSource]


class PipelineRunner:
    """Runs one job end to end. All collaborators are injected, so tests can fake each one."""

    def __init__(
        self,
        sources: SourceFactory,
        detectors: DetectorFactory,
        repository: JobRepository,
        batch_frames: int = 32,
    ) -> None:
        self._sources = sources
        self._detectors = detectors
        self._repository = repository
        self._batch_frames = batch_frames

    def run(self, job: Job) -> None:
        self._repository.set_status(job.id, JobStatus.RUNNING)
        try:
            detector = self._detectors(job.detector)
            batch: list[tuple[int, float, Frame]] = []
            for item in self._sources(job).frames():
                batch.append(item)
                if len(batch) >= self._batch_frames:
                    self._process(job.id, detector, batch)
                    batch = []
            self._process(job.id, detector, batch)
        except Exception as exc:
            log.exception("job failed", extra={"job_id": job.id})
            self._repository.set_status(
                job.id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
            return
        self._repository.set_status(job.id, JobStatus.DONE)

    def _process(
        self, job_id: str, detector: Detector, batch: list[tuple[int, float, Frame]]
    ) -> None:
        if not batch:
            return
        results = detector.detect_batch([frame for _, _, frame in batch])
        frames = [FrameDetections(i, t, d) for (i, t, _), d in zip(batch, results, strict=True)]
        self._repository.save_detections(job_id, frames)
