"""Synthetic test media with known ground truth.

A white square sits still, moves right during a known window, then stops.
Motion therefore happens exactly on frames MOTION_START..MOTION_END.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FPS = 10.0
WIDTH, HEIGHT = 160, 120
FRAMES = 40
MOTION_START, MOTION_END = 10, 29  # inclusive frame indices with motion
SQUARE = 20
STEP = 4  # pixels per frame while moving


def square_x(index: int) -> int:
    moving_frames = min(max(index - (MOTION_START - 1), 0), MOTION_END - MOTION_START + 1)
    return 10 + STEP * moving_frames


def write_moving_square(path: Path, frames: int = FRAMES) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write MJPG video on this machine")
    for i in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x = square_x(i)
        frame[50 : 50 + SQUARE, x : x + SQUARE] = 255
        writer.write(frame)
    writer.release()
    return path


def write_still(path: Path, frames: int = 20) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    frame = np.full((HEIGHT, WIDTH, 3), 80, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return path


def write_corrupt(path: Path) -> Path:
    path.write_bytes(b"this is not a video" * 100)
    return path


def write_image_clip(path: Path, image_path: Path, start: int = 10, end: int = 29) -> Path:
    """A clip that shows `image_path` on frames start..end (inclusive) and black elsewhere."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (w, h))
    black = np.zeros_like(image)
    for i in range(FRAMES):
        writer.write(image if start <= i <= end else black)
    writer.release()
    return path
