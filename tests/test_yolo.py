"""YOLO26 processing: hand-computed cases first, then the real model, then Triton over HTTP."""

from __future__ import annotations

import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from checks.fake_triton import FakeTriton
from shared.detection import Detection
from shared.jobs import InMemoryJobRepository, Job, JobStatus
from shared.model_detectors import DetectorUnavailableError, OnnxDetector, TritonDetector, YoloIO
from shared.pipeline import PipelineRunner
from shared.video import Frame
from shared.yolo import Placement, iou, letterbox, nms, to_detections

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models/yolo26n.onnx"
IMAGE = REPO / "checks/data/astronaut.jpg"  # public domain, NASA; via scikit-image's sample data
IO = YoloIO((640, 640), score_threshold=0.3, nms_threshold=0.45)

needs_model = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"{MODEL.name} missing: see README 'Models' (export with a dynamic batch)",
)


def load_image() -> Frame:
    image = cv2.imread(str(IMAGE))
    assert image is not None, f"cannot read {IMAGE}"
    return np.asarray(image, dtype=np.uint8)


# ------------------------------------------------------------------ pure functions


def test_letterbox_centres_converts_to_rgb_and_scales() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200, blue in BGR
    frame[:, :, 0] = 255
    tensor, placement = letterbox(frame, (50, 50))
    assert placement == Placement(ratio=0.25, pad_x=0, pad_y=12, height=100, width=200)
    assert tensor.shape == (3, 50, 50) and tensor.dtype == np.float32
    image = tensor[:, 12:37, :]  # 100 * 0.25 = 25 rows of image, centred: 12 above, 13 below
    assert (image[2] == 1.0).all() and (image[0] == 0.0).all()  # blue lands in channel 2 (RGB)
    assert np.allclose(tensor[:, :12, :], 114 / 255) and np.allclose(tensor[:, 37:, :], 114 / 255)


def test_read_rejects_an_output_of_the_wrong_shape() -> None:
    placement = Placement(1.0, 0, 0, 10, 10)
    with pytest.raises(ValueError, match="84"):
        IO.read(np.zeros((1, 85, 10), dtype=np.float32), [placement])  # a YOLOX-style output


def test_iou_and_nms() -> None:
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    others = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], dtype=np.float32)
    assert iou(a, others).tolist() == pytest.approx([1.0, 0.0, 1 / 3])
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.5) == [0, 2]


def row(cx: float, cy: float, cls: list[float]) -> list[float]:
    return [cx, cy, 20, 20, *cls]


def test_to_detections_thresholds_unpads_scales_and_suppresses_per_class() -> None:
    classes = ("a", "b", "c")
    rows = np.array(
        [
            row(60, 50, [0.9, 0, 0]),  # kept: class a, score 0.9
            row(62, 50, [0.8, 0, 0]),  # suppressed: overlaps the first, same class
            row(60, 50, [0, 0.7, 0]),  # kept: same place, different class
            row(200, 200, [0.2, 0, 0]),  # dropped: below the score threshold
        ],
        dtype=np.float32,
    )
    placement = Placement(ratio=2.0, pad_x=10, pad_y=0, height=100, width=100)
    found = to_detections(rows, placement, score_threshold=0.3, nms_threshold=0.45, classes=classes)
    assert [(d.label, round(d.score, 2)) for d in found] == [("a", 0.9), ("b", 0.7)]
    box = found[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == (20, 20, 30, 30)  # (50..70 - pad 10) / 2
    only_b = to_detections(
        rows, placement, 0.3, 0.45, classes=classes, keep_labels=frozenset({"b"})
    )
    assert [d.label for d in only_b] == ["b"]


def test_boxes_are_clipped_to_the_frame() -> None:
    rows = np.array([row(5, 5, [0.9])], dtype=np.float32)  # spans -5..15
    [found] = to_detections(rows, Placement(1.0, 0, 0, 10, 10), 0.3, 0.45, classes=("a",))
    assert (found.box.x1, found.box.y1, found.box.x2, found.box.y2) == (0, 0, 10, 10)


# ------------------------------------------------------------------ the real model


def labels(results: list[list[Detection]]) -> list[list[str]]:
    return [[d.label for d in frame] for frame in results]


@needs_model
def test_onnx_finds_the_astronaut() -> None:
    image = load_image()
    [found] = OnnxDetector(MODEL, IO).detect_batch([image])
    people = [d for d in found if d.label == "person"]
    assert len(people) == 1
    # Ultralytics' own predict() on this model and image gives person 0.849 at
    # [12.5, 16.1, 362.3, 511.9]: matching it shows the pre- and post-processing agree.
    b = people[0].box
    assert people[0].score == pytest.approx(0.849, abs=0.01)
    assert [b.x1, b.y1, b.x2, b.y2] == pytest.approx([12.5, 16.1, 362.3, 511.9], abs=2)


@needs_model
def test_batches_give_the_same_answer_as_single_frames() -> None:
    image = load_image()
    flipped = np.ascontiguousarray(image[:, ::-1])
    detector = OnnxDetector(MODEL, IO, max_batch=8)
    batch = detector.detect_batch([image, flipped, image])
    singles = [detector.detect_batch([f])[0] for f in (image, flipped, image)]
    assert batch == singles
    # The mirror image gives roughly the mirrored box (a CNN is not exactly flip-symmetric).
    left, right = batch[0][0].box, batch[1][0].box
    assert abs((512 - left.x2) - right.x1) < 5 and abs((512 - left.x1) - right.x2) < 5


@needs_model
def test_triton_detector_matches_local_onnx_and_batches_requests() -> None:
    image = load_image()
    frames = [image] * 5
    with FakeTriton(MODEL) as server:
        remote = TritonDetector(server.url, "yolo26n", IO, max_batch=4, timeout_s=10)
        assert remote.detect_batch(frames) == OnnxDetector(MODEL, IO).detect_batch(frames)
        assert server.batch_sizes == [4, 1]  # 5 frames, max batch 4: two requests, not five


@needs_model
def test_triton_retries_then_succeeds() -> None:
    image = load_image()
    with FakeTriton(MODEL, fail_first=2) as server:
        found = TritonDetector(server.url, "yolo26n", IO, retries=3, timeout_s=10).detect_batch(
            [image]
        )
        assert labels(found) == [["person"]]
        assert server.requests == 3


def closed_port() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"  # nothing listens here once the socket closes


def test_triton_down_fails_the_job_with_a_readable_error() -> None:
    detector = TritonDetector(closed_port(), "yolo26n", IO, retries=2, timeout_s=2)
    with pytest.raises(DetectorUnavailableError, match="failed 2 times"):
        detector.detect_batch([np.zeros((64, 64, 3), dtype=np.uint8)])

    class OneFrame:
        def frames(self):
            yield 0, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

    repo = InMemoryJobRepository()
    job = Job(video_id="v", video_path="unused", detector="triton")
    repo.add(job)
    PipelineRunner(lambda _j: OneFrame(), lambda _n: detector, repo).run(job)
    failed = repo.get(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and "Triton at" in failed.error


@needs_model
def test_triton_config_matches_the_model() -> None:
    import re

    import onnxruntime as ort

    config = (REPO / "model_repository/yolo26n/config.pbtxt").read_text()
    dims = [[int(v) for v in d.split(",")] for d in re.findall(r"dims: \[([^\]]+)\]", config)]
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    assert isinstance(model_input.shape[0], str), "Triton batching needs a dynamic batch model"
    # Triton dims leave out the batch; run the model at the configured size to compare.
    out = session.run(None, {model_input.name: np.zeros([2, *dims[0]], dtype=np.float32)})[0]
    assert list(np.asarray(out).shape) == [2, *dims[1]]
