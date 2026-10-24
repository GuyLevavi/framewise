"""YOLO26 input and output handling. Plain functions: arrays in, arrays out.

Matches Ultralytics' own pre- and post-processing for a YOLO26 ONNX export: the frame is
letterboxed into the centre of the input with grey (114) padding, converted BGR to RGB and
scaled to 0..1. The output is (batch, 4 + classes, candidates): centre-x, centre-y, width,
height in input pixels, then one probability per class. There is no objectness score.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from shared.detection import Box, Detection
from shared.video import Frame

FloatArray = NDArray[np.float32]

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip


class Placement(NamedTuple):
    """Where the frame landed inside the model input, to map boxes back."""

    ratio: float
    pad_x: int
    pad_y: int
    height: int  # original frame size
    width: int


def letterbox(frame: Frame, size: tuple[int, int]) -> tuple[FloatArray, Placement]:
    """Resize keeping aspect ratio, centre on grey padding, BGR to RGB, scale to 0..1 (CHW)."""
    h, w = size
    fh, fw = frame.shape[:2]
    ratio = min(h / fh, w / fw)
    new_w, new_h = round(fw * ratio), round(fh * ratio)
    pad_x, pad_y = round((w - new_w) / 2 - 0.1), round((h - new_h) / 2 - 0.1)
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255)
    return tensor, Placement(ratio, pad_x, pad_y, fh, fw)


def iou(box: FloatArray, others: FloatArray) -> FloatArray:
    """Intersection over union of one xyxy box with each row of `others`."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: FloatArray, scores: FloatArray, threshold: float) -> list[int]:
    """Indices of boxes kept by greedy non-maximum suppression, best score first."""
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        best, rest = order[0], np.array(order[1:], dtype=np.int64)
        keep.append(int(best))
        if rest.size == 0:
            break
        order = [int(i) for i in rest[iou(boxes[best], boxes[rest]) <= threshold]]
    return keep


def to_detections(
    rows: FloatArray,
    placement: Placement,
    score_threshold: float,
    nms_threshold: float,
    classes: tuple[str, ...] = COCO_CLASSES,
    keep_labels: frozenset[str] | None = None,
) -> list[Detection]:
    """One image's candidates (A, 4 + C) to detections in original-frame pixels."""
    class_scores = rows[:, 4:]
    best = class_scores.argmax(axis=1)
    score = class_scores[np.arange(len(best)), best]
    mask = score >= score_threshold
    if keep_labels is not None:
        mask &= np.isin(best, [classes.index(label) for label in keep_labels])
    cx, cy, w, h = (rows[mask, i] for i in range(4))
    xyxy = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)
    xyxy -= np.array([placement.pad_x, placement.pad_y] * 2, dtype=np.float32)
    xyxy /= placement.ratio
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, placement.width)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, placement.height)
    boxes = xyxy.astype(np.float32)
    labels, scores = best[mask], score[mask]
    detections: list[Detection] = []
    for label in np.unique(labels):  # suppress overlaps within a class, not across classes
        idx = np.nonzero(labels == label)[0]
        for k in nms(boxes[idx], scores[idx], nms_threshold):
            x1, y1, x2, y2 = (float(v) for v in boxes[idx[k]])
            detections.append(
                Detection(classes[int(label)], float(scores[idx[k]]), Box(x1, y1, x2, y2))
            )
    return sorted(detections, key=lambda d: -d.score)
