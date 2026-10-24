"""Week 6 acceptance: `pytest -m eval` passes a good model and blocks a bad setting.

Contract: the gate reads its settings from FRAMEWISE_EVAL_CONFIG (schema: eval/eval.yaml).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from checks import synth
from checks.fixtures import REPO

MODEL = REPO / "models/yolo26n.onnx"
IMAGE = REPO / "checks/data/astronaut.jpg"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} missing (see README: Models)")


def run_gate(tmp_path: Path, detector_overrides: str) -> subprocess.CompletedProcess[str]:
    data = tmp_path / "labels"
    data.mkdir()
    video = synth.write_image_clip(data / "astronaut.avi", IMAGE, start=10, end=29)
    synth.write_label_studio_export(data / "astronaut.json", video, "person", 10, 29)
    detectors = tmp_path / "detectors.yaml"
    detectors.write_text(
        "sample_fps: 10\nmax_gap_s: 0.5\ndetectors:\n"
        f"  under-test:\n    kind: onnx\n    model_path: {MODEL}\n{detector_overrides}"
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"detectors_config: {detectors}\ndetector: under-test\nlabels_dir: {data}\n"
        "thresholds:\n  person: {precision: 0.9, recall: 0.9}\n"
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "eval",
            "-s",
            "-q",
            "-p",
            "no:cacheprovider",
            "eval",
        ],
        cwd=REPO,
        env={**os.environ, "FRAMEWISE_EVAL_CONFIG": str(config)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_gate_passes_a_good_model_and_reports_versions(tmp_path: Path) -> None:
    result = run_gate(tmp_path, "    score_threshold: 0.3\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "model=" in result.stdout and "data=" in result.stdout, "log model and data versions"


def test_gate_blocks_a_bad_setting(tmp_path: Path) -> None:
    result = run_gate(tmp_path, "    score_threshold: 0.95\n")  # finds almost nothing
    assert result.returncode != 0, result.stdout
    assert "person" in result.stdout
