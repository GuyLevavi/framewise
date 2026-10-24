"""Scoring a detector against labeled clips, frame by frame.

Where labels come from is behind `LabelSource`. `LocalLabelSource` reads Label Studio exports
from a folder; the team's helper library (DagsHub + Label Studio) plugs in as another
`LabelSource` with the same `clips()` method.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

FrameLabels = dict[int, set[str]]  # frame index (0-based) -> labels present in that frame


@dataclass(frozen=True)
class LabeledClip:
    name: str
    video_path: Path
    frames: FrameLabels  # frames not listed contain none of the labeled classes


@dataclass(frozen=True)
class ClassScore:
    label: str
    true_pos: int
    false_pos: int
    false_neg: int

    @property
    def precision(self) -> float:
        found = self.true_pos + self.false_pos
        return self.true_pos / found if found else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_pos + self.false_neg
        return self.true_pos / actual if actual else 1.0


class LabelSource(Protocol):
    def clips(self) -> Iterator[LabeledClip]: ...

    def version(self) -> str:
        """Identifies the exact label data, so a change in scores can be traced to it."""
        ...


def frames_from_label_studio(task: dict[str, object]) -> FrameLabels:
    """Frame labels from one Label Studio video task with TimelineLabels annotations.

    Label Studio numbers video frames from 1; Framewise numbers them from 0.
    """
    frames: FrameLabels = {}
    annotations = task.get("annotations") or []
    assert isinstance(annotations, list)
    for annotation in annotations:
        for result in annotation["result"]:
            if result.get("type") != "timelinelabels":
                continue
            value = result["value"]
            for frame_range in value["ranges"]:
                for frame in range(int(frame_range["start"]) - 1, int(frame_range["end"])):
                    frames.setdefault(frame, set()).update(value["timelinelabels"])
    return frames


class LocalLabelSource:
    """A folder of Label Studio JSON exports, each next to the video it labels."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder

    def _exports(self) -> list[Path]:
        exports = sorted(self._folder.glob("*.json"))
        if not exports:
            raise FileNotFoundError(f"no Label Studio exports (*.json) in {self._folder}")
        return exports

    def clips(self) -> Iterator[LabeledClip]:
        for export in self._exports():
            for task in json.loads(export.read_text()):
                video = self._folder / Path(str(task["data"]["video"])).name
                yield LabeledClip(video.stem, video, frames_from_label_studio(task))

    def version(self) -> str:
        digest = hashlib.sha256()
        for export in self._exports():
            digest.update(export.read_bytes())
        return f"local:{digest.hexdigest()[:12]}"


def score_frames(predicted: FrameLabels, truth: FrameLabels, labels: list[str]) -> list[ClassScore]:
    """Per label, count frames found correctly, found wrongly and missed. Only predicted frames
    (the frames the pipeline sampled) are scored."""
    scores: list[ClassScore] = []
    for label in labels:
        tp = fp = fn = 0
        for frame, found in predicted.items():
            is_there, said_there = label in truth.get(frame, set()), label in found
            tp += is_there and said_there
            fp += said_there and not is_there
            fn += is_there and not said_there
        scores.append(ClassScore(label, tp, fp, fn))
    return scores


def add_scores(a: list[ClassScore], b: list[ClassScore]) -> list[ClassScore]:
    return [
        ClassScore(
            x.label, x.true_pos + y.true_pos, x.false_pos + y.false_pos, x.false_neg + y.false_neg
        )
        for x, y in zip(a, b, strict=True)
    ]


def file_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
