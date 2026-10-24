"""YAML configuration, validated by Pydantic. Variants are data, not classes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.detection import Detector, MotionDetector
from shared.model_detectors import OnnxDetector, TritonDetector, YoloIO
from shared.pipeline import DetectorFactory


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotionConfig(_Strict):
    kind: Literal["motion"]
    threshold: int = Field(default=25, ge=1, le=255)
    min_area: int = Field(default=50, ge=1)


class _YoloConfig(_Strict):
    input_size: tuple[int, int] = (640, 640)
    score_threshold: float = Field(default=0.3, gt=0, lt=1)
    nms_threshold: float = Field(default=0.45, gt=0, lt=1)
    keep_labels: list[str] | None = None  # e.g. [person, car]; None keeps all 80 COCO classes
    max_batch: int = Field(default=8, ge=1)

    def io(self) -> YoloIO:
        keep = frozenset(self.keep_labels) if self.keep_labels else None
        return YoloIO(self.input_size, self.score_threshold, self.nms_threshold, keep)


class OnnxConfig(_YoloConfig):
    kind: Literal["onnx"]
    model_path: Path


class TritonConfig(_YoloConfig):
    kind: Literal["triton"]
    url: str  # host:port of Triton's HTTP endpoint, no scheme
    model_name: str = "yolo26n"
    timeout_s: float = Field(default=30, gt=0)
    retries: int = Field(default=3, ge=1)


DetectorConfig = Annotated[MotionConfig | OnnxConfig | TritonConfig, Field(discriminator="kind")]


class DetectorsFile(_Strict):
    sample_fps: float = Field(default=5.0, gt=0)
    max_gap_s: float = Field(default=0.5, ge=0)
    detectors: dict[str, DetectorConfig]


def load_yaml[M: BaseModel](path: Path, model: type[M]) -> M:
    with path.open(encoding="utf-8") as fh:
        return model.model_validate(yaml.safe_load(fh))


def build_detector(config: DetectorConfig) -> Detector:
    match config:
        case MotionConfig():
            return MotionDetector(threshold=config.threshold, min_area=config.min_area)
        case OnnxConfig():
            return OnnxDetector(config.model_path, config.io(), config.max_batch)
        case TritonConfig():
            return TritonDetector(
                config.url,
                config.model_name,
                config.io(),
                max_batch=config.max_batch,
                timeout_s=config.timeout_s,
                retries=config.retries,
            )


def detector_factory(file: DetectorsFile) -> DetectorFactory:
    def make(name: str) -> Detector:
        if name not in file.detectors:
            raise KeyError(f"unknown detector {name!r}; known: {sorted(file.detectors)}")
        return build_detector(file.detectors[name])

    return make
