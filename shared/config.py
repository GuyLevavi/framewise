"""YAML configuration, validated by Pydantic. Variants are data, not classes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.detection import Detector, MotionDetector
from shared.pipeline import DetectorFactory


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotionConfig(_Strict):
    kind: Literal["motion"]
    threshold: int = Field(default=25, ge=1, le=255)
    min_area: int = Field(default=50, ge=1)


DetectorConfig = Annotated[MotionConfig, Field(discriminator="kind")]


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


def detector_factory(file: DetectorsFile) -> DetectorFactory:
    def make(name: str) -> Detector:
        if name not in file.detectors:
            raise KeyError(f"unknown detector {name!r}; known: {sorted(file.detectors)}")
        return build_detector(file.detectors[name])

    return make
