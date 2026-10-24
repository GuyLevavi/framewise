"""YOLO detectors: in-process with ONNX Runtime, or remote on Triton. Same inputs, same outputs."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from shared.detection import Detection
from shared.video import Frame
from shared.yolo import COCO_CLASSES, FloatArray, Placement, letterbox, to_detections

log = logging.getLogger(__name__)


class DetectorUnavailableError(RuntimeError):
    """The model server could not be reached or kept failing."""


@dataclass(frozen=True)
class YoloIO:
    """The model-independent half: prepare a batch, read the raw output back. Settings only."""

    input_size: tuple[int, int]
    score_threshold: float = 0.3
    nms_threshold: float = 0.45
    keep_labels: frozenset[str] | None = None

    def prepare(self, frames: Sequence[Frame]) -> tuple[FloatArray, list[Placement]]:
        tensors, placements = zip(*(letterbox(f, self.input_size) for f in frames), strict=True)
        return np.stack(tensors), list(placements)

    def read(self, raw: FloatArray, placements: list[Placement]) -> list[list[Detection]]:
        """`raw` is (batch, 4 + classes, candidates), as the model returns it."""
        if raw.ndim != 3 or raw.shape[1] != 4 + len(COCO_CLASSES):
            raise ValueError(
                f"expected (batch, {4 + len(COCO_CLASSES)}, N) output, got {raw.shape}"
            )
        rows = raw.transpose(0, 2, 1)  # one row per candidate box
        return [
            to_detections(
                r, p, self.score_threshold, self.nms_threshold, keep_labels=self.keep_labels
            )
            for r, p in zip(rows, placements, strict=True)
        ]


class OnnxDetector:
    """Runs the model in this process on the CPU. Needs a dynamic-batch model for batches > 1."""

    def __init__(self, model_path: Path, io: YoloIO, max_batch: int = 8) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._io = io
        batch_dim = self._session.get_inputs()[0].shape[0]
        self._max_batch = max_batch if isinstance(batch_dim, str) else 1

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for start in range(0, len(frames), self._max_batch):
            chunk = frames[start : start + self._max_batch]
            tensor, placements = self._io.prepare(chunk)
            raw = np.asarray(self._session.run(None, {self._input: tensor})[0], dtype=np.float32)
            results.extend(self._io.read(raw, placements))
        return results


class TritonDetector:
    """Sends batches to Triton over HTTP with binary tensors; retries a few times, then gives up."""

    def __init__(
        self,
        url: str,
        model_name: str,
        io: YoloIO,
        input_name: str = "images",
        output_name: str = "output0",
        max_batch: int = 8,
        timeout_s: float = 30.0,
        retries: int = 3,
    ) -> None:
        import tritonclient.http as httpclient  # imported here: optional for CPU-only use

        self._http = httpclient
        self._client = httpclient.InferenceServerClient(
            url=url, connection_timeout=timeout_s, network_timeout=timeout_s
        )
        self._url, self._model = url, model_name
        self._input_name, self._output_name = input_name, output_name
        self._io, self._max_batch, self._retries = io, max_batch, retries

    def detect_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for start in range(0, len(frames), self._max_batch):
            tensor, placements = self._io.prepare(frames[start : start + self._max_batch])
            results.extend(self._io.read(self._infer(tensor), placements))
        return results

    def _infer(self, tensor: FloatArray) -> FloatArray:
        request = self._http.InferInput(self._input_name, list(tensor.shape), "FP32")
        request.set_data_from_numpy(tensor, binary_data=True)  # raw bytes, not JSON numbers
        wanted = self._http.InferRequestedOutput(self._output_name, binary_data=True)
        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = self._client.infer(self._model, inputs=[request], outputs=[wanted])
                output = response.as_numpy(self._output_name)
                if output is None:
                    raise DetectorUnavailableError(f"Triton returned no {self._output_name!r}")
                return np.asarray(output, dtype=np.float32)
            except Exception as exc:
                last_error = exc
                log.warning("triton attempt %d/%d failed: %s", attempt, self._retries, exc)
                time.sleep(0.2 * 2 ** (attempt - 1))
        raise DetectorUnavailableError(
            f"Triton at {self._url} (model {self._model}) failed {self._retries} times: "
            f"{last_error}"
        )
