"""Week 5 acceptance: YOLO26n finds people, locally and through Triton's HTTP protocol."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from checks import synth
from checks.fake_triton import FakeTriton
from checks.fixtures import REPO, poll_job, write_api_config, write_worker_config
from checks.test_week03 import upload

MODEL = REPO / "models/yolo26n.onnx"
IMAGE = REPO / "checks/data/astronaut.jpg"
DETECTORS = REPO / "worker/configs/detectors.yaml"
ONE_FRAME = 1 / synth.FPS

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL} missing (see README: Models)")


def summary_after_job(url: str, tmp_path: Path, detector: str) -> list[dict[str, object]]:
    video_id = upload(url, synth.write_image_clip(tmp_path / "astronaut.avi", IMAGE))
    r = httpx.post(f"{url}/jobs", json={"video_id": video_id, "detector": detector})
    assert r.status_code == 202, r.text
    job = poll_job(url, str(r.json()["id"]), timeout_s=120)
    assert job["status"] == "DONE", job
    s = httpx.get(f"{url}/videos/{video_id}/summary")
    assert s.status_code == 200, s.text
    return list(s.json())


def assert_person_in_window(summary: list[dict[str, object]]) -> None:
    people = [row for row in summary if row["label"] == "person"]
    assert people, f"no person found: {summary}"
    first, last = float(str(people[0]["first_seen_s"])), float(str(people[0]["last_seen_s"]))
    assert abs(first - 10 / synth.FPS) <= ONE_FRAME + 1e-6
    assert abs(last - 29 / synth.FPS) <= ONE_FRAME + 1e-6


def test_local_onnx_detector_finds_the_person(api_url: str, tmp_path: Path) -> None:
    assert_person_in_window(summary_after_job(api_url, tmp_path, "people-cars-local"))


@pytest.fixture
def fake_triton() -> Iterator[FakeTriton]:
    with FakeTriton(MODEL) as server:
        yield server


@pytest.fixture
def app_env(tmp_path: Path, clean_db: str, fake_triton: FakeTriton) -> dict[str, str]:
    """Like the default environment, but the triton detector points at the fake server."""
    detectors = tmp_path / "detectors.yaml"
    detectors.write_text(
        DETECTORS.read_text()
        .replace("url: localhost:8000", f"url: {fake_triton.url}")
        .replace("model_path: models/", f"model_path: {REPO}/models/")
    )
    configs = (write_api_config(tmp_path), write_worker_config(tmp_path))
    for config in configs:
        config.write_text(config.read_text().replace(str(DETECTORS), str(detectors)))
    return {
        "FRAMEWISE_API_CONFIG": str(configs[0]),
        "FRAMEWISE_WORKER_CONFIG": str(configs[1]),
        "DATABASE_URL": clean_db,
    }


def test_triton_detector_finds_the_person_and_batches(
    api_url: str, tmp_path: Path, fake_triton: FakeTriton
) -> None:
    assert_person_in_window(summary_after_job(api_url, tmp_path, "people-cars-triton"))
    assert max(fake_triton.batch_sizes) > 1, "frames were sent one per request"


def test_unknown_video_summary_is_404(api_url: str) -> None:
    assert httpx.get(f"{api_url}/videos/nope/summary").status_code == 404
