from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings
from checks import synth
from shared.jobs import InMemoryJobRepository

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = ApiSettings(
        upload_dir=tmp_path / "uploads",
        detectors_config=REPO / "worker/configs/detectors.yaml",
    )
    with TestClient(create_app(settings, InMemoryJobRepository())) as c:
        yield c


def wait_done(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        body: dict[str, object] = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("DONE", "FAILED"):
            return body
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_full_flow(client: TestClient, tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "v.avi")
    with video.open("rb") as fh:
        video_id = client.post("/videos", files={"file": ("v.avi", fh)}).json()["video_id"]
    job = client.post("/jobs", json={"video_id": video_id}).json()
    assert job["status"] == "PENDING"
    done = wait_done(client, job["id"])
    assert done["status"] == "DONE"
    assert done["segments"] == [{"start_s": 1.0, "end_s": 2.9}]


def test_validation_error_is_422(client: TestClient) -> None:
    assert client.post("/jobs", json={"detector": "x"}).status_code == 422
