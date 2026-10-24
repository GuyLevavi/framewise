"""The evaluation gate: `pytest -m eval`. CI runs it before building an image.

It runs the production PipelineRunner and detector over labeled clips and fails when any
class drops below its threshold in eval/eval.yaml (or the file named by FRAMEWISE_EVAL_CONFIG).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from shared.config import DetectorsFile, OnnxConfig, detector_factory, load_yaml
from shared.evaluation import (
    ClassScore,
    FrameLabels,
    LabelSource,
    LocalLabelSource,
    add_scores,
    file_version,
    score_frames,
)
from shared.jobs import InMemoryJobRepository, Job
from shared.pipeline import PipelineRunner, VideoFileSource

CONFIG_ENV = "FRAMEWISE_EVAL_CONFIG"


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detectors_config: Path
    detector: str
    labels_dir: Path  # LocalLabelSource; swap for the team helper library's source
    thresholds: dict[str, Threshold]


def label_source(config: EvalConfig) -> LabelSource:
    return LocalLabelSource(config.labels_dir)


def model_version(detectors: DetectorsFile, name: str) -> str:
    detector = detectors.detectors[name]
    if isinstance(detector, OnnxConfig):
        return f"{detector.model_path.name}:{file_version(detector.model_path)}"
    return f"{name}:{detector.model_dump_json()}"


@pytest.mark.eval
def test_detector_meets_thresholds_on_labeled_clips() -> None:
    config = load_yaml(Path(os.environ.get(CONFIG_ENV, "eval/eval.yaml")), EvalConfig)
    detectors = load_yaml(config.detectors_config, DetectorsFile)
    source = label_source(config)
    labels = sorted(config.thresholds)
    total = [ClassScore(label, 0, 0, 0) for label in labels]
    for clip in source.clips():
        repo = InMemoryJobRepository()
        runner = PipelineRunner(
            lambda job: VideoFileSource(Path(job.video_path), detectors.sample_fps),
            detector_factory(detectors),
            repo,
        )
        job = Job(video_id=clip.name, video_path=str(clip.video_path), detector=config.detector)
        repo.add(job)
        runner.run(job)
        assert repo.get(job.id).error is None, f"{clip.name}: {repo.get(job.id).error}"
        predicted: FrameLabels = {}
        for index, _t, _f in VideoFileSource(clip.video_path, detectors.sample_fps).frames():
            predicted[index] = set()
        for frame in repo.list_detections(job.id):
            predicted[frame.frame_index] = {d.label for d in frame.detections}
        total = add_scores(total, score_frames(predicted, clip.frames, labels))

    print(f"\nmodel={model_version(detectors, config.detector)} data={source.version()}")
    failures: list[str] = []
    for s in total:
        want = config.thresholds[s.label]
        counts = f"tp={s.true_pos} fp={s.false_pos} fn={s.false_neg}"
        print(f"{s.label:>10}  precision={s.precision:.3f}  recall={s.recall:.3f}  ({counts})")
        if s.precision < want.precision or s.recall < want.recall:
            failures.append(
                f"{s.label}: precision {s.precision:.3f} (min {want.precision}),"
                f" recall {s.recall:.3f} (min {want.recall})"
            )
    assert not failures, "below threshold: " + "; ".join(failures)
