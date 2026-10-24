"""Detectors: one interface, a few implementations, variants chosen by config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from shared.video import Frame


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box


class Detector(Protocol):
    """Anything that turns one BGR frame into detections."""

    def detect(self, frame: Frame) -> list[Detection]: ...


class MotionDetector:
    """Boxes pixels that changed since the previous frame.

    A class because it keeps state (the previous frame). Different thresholds are
    configuration, not subclasses: see worker/configs/detectors.yaml.
    """

    def __init__(self, threshold: int = 25, min_area: int = 50) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._previous: Frame | None = None

    def detect(self, frame: Frame) -> list[Detection]:
        gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return []
        diff = cv2.absdiff(gray, previous)
        mask = (diff > self._threshold).astype(np.uint8)
        if int(mask.sum()) < self._min_area:
            return []
        ys, xs = np.nonzero(mask)
        box = Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        return [Detection(label="motion", score=1.0, box=box)]
