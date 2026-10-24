"""Reading video files: metadata and sampled frames."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class VideoError(Exception):
    """The file is missing, unreadable, or not a video OpenCV can decode."""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def open_video(path: Path) -> Iterator[cv2.VideoCapture]:
    """Open a video and always release it, even if the caller raises."""
    if not path.is_file():
        raise VideoError(f"no such file: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        ok, _ = capture.read()
        if not capture.isOpened() or not ok:
            raise VideoError(f"cannot decode video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        yield capture
    finally:
        capture.release()


def probe(path: Path) -> VideoInfo:
    with open_video(path) as capture:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(
            path=str(path),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=frame_count,
            duration_s=round(frame_count / fps, 3) if fps > 0 else 0.0,
        )


def iter_frames(path: Path, every_n: int = 1) -> Iterator[tuple[int, Frame]]:
    """Yield (frame_index, BGR frame) for every n-th frame."""
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    with open_video(path) as capture:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if index % every_n == 0:
                yield index, np.asarray(frame, dtype=np.uint8)
            index += 1


def save_frames(path: Path, out_dir: Path, every_n: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, frame in iter_frames(path, every_n):
        target = out_dir / f"frame_{index:06d}.jpg"
        if not cv2.imwrite(str(target), frame):
            raise VideoError(f"cannot write {target}")
        saved.append(target)
    return saved
