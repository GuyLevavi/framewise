from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from checks import synth
from shared.config import DetectorsFile, MotionConfig, build_detector, detector_factory
from shared.detection import Box, Detection, Detector, MotionDetector
from shared.jobs import InMemoryJobRepository, Job, JobStatus
from shared.pipeline import FrameSource, PipelineRunner, Segment, VideoFileSource, group_segments
from shared.video import Frame


class FakeDetector:
    """Reports one box on every frame whose mean brightness is above 0."""

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        return [[Detection("thing", 0.9, Box(0, 0, 1, 1))] if f.mean() > 0 else [] for f in frames]


class ListSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[tuple[int, float, Frame]]:
        for i, f in enumerate(self._frames):
            yield i, i / 10, f


class ExplodingSource:
    def frames(self) -> Iterator[tuple[int, float, Frame]]:
        yield 0, 0.0, np.zeros((4, 4, 3), dtype=np.uint8)
        raise RuntimeError("decoder crashed")


def make_runner(
    source: FrameSource, detector: Detector
) -> tuple[PipelineRunner, InMemoryJobRepository]:
    repo = InMemoryJobRepository()
    runner = PipelineRunner(lambda _job: source, lambda _name: detector, repo, batch_frames=2)
    return runner, repo


def test_runner_stores_detections_for_every_frame_and_finishes() -> None:
    frames = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (0, 9, 0, 9, 9)]
    runner, repo = make_runner(ListSource(frames), FakeDetector())
    job = Job(video_id="v", video_path="unused", detector="fake")
    repo.add(job)
    runner.run(job)
    assert repo.get(job.id).status is JobStatus.DONE
    stored = repo.list_detections(job.id)
    assert [f.frame_index for f in stored] == [1, 3, 4]  # frames without detections are not stored


def test_runner_marks_job_failed_with_reason() -> None:
    runner, repo = make_runner(ExplodingSource(), FakeDetector())
    job = Job(video_id="v", video_path="unused", detector="fake")
    repo.add(job)
    runner.run(job)
    failed = repo.get(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and "decoder crashed" in failed.error


def test_motion_detector_finds_exactly_the_moving_window(tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "v.avi")
    detector = MotionDetector(threshold=15, min_area=20)
    moving = [i for i, _t, f in VideoFileSource(video, synth.FPS).frames() if detector.detect(f)]
    assert moving[0] in range(synth.MOTION_START - 1, synth.MOTION_START + 2)
    assert moving[-1] in range(synth.MOTION_END - 1, synth.MOTION_END + 2)
    assert len(moving) == pytest.approx(synth.MOTION_END - synth.MOTION_START + 1, abs=2)


def test_group_segments_merges_close_times_and_splits_on_gaps() -> None:
    assert group_segments([0.0, 0.1, 0.2, 1.0, 1.1], max_gap_s=0.3) == [
        Segment(0.0, 0.2),
        Segment(1.0, 1.1),
    ]
    assert group_segments([], max_gap_s=0.3) == []


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DetectorsFile.model_validate(
            {"detectors": {"m": {"kind": "motion", "treshold": 10}}}  # typo on purpose
        )


def test_variants_are_config_not_classes() -> None:
    file = DetectorsFile.model_validate(
        {
            "detectors": {
                "a": {"kind": "motion", "threshold": 10},
                "b": {"kind": "motion", "threshold": 80},
            }
        }
    )
    make = detector_factory(file)
    assert type(make("a")) is type(make("b")) is MotionDetector
    with pytest.raises(KeyError, match="unknown detector"):
        make("c")
    assert isinstance(build_detector(MotionConfig(kind="motion")), MotionDetector)
