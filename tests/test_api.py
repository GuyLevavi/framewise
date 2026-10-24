from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings
from checks import synth
from shared.config import DetectorsFile, detector_factory, load_yaml
from shared.jobs import InMemoryJobRepository
from shared.pipeline import PipelineRunner, VideoFileSource

REPO = Path(__file__).resolve().parent.parent
DETECTORS = REPO / "worker/configs/detectors.yaml"


@pytest.fixture
def repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def client(tmp_path: Path, repo: InMemoryJobRepository) -> Iterator[TestClient]:
    settings = ApiSettings(upload_dir=tmp_path / "uploads", detectors_config=DETECTORS)
    with TestClient(create_app(settings, repo)) as c:
        yield c


def run_one_job(repo: InMemoryJobRepository) -> None:
    """Stand in for the worker process."""
    detectors = load_yaml(DETECTORS, DetectorsFile)
    runner = PipelineRunner(
        lambda job: VideoFileSource(Path(job.video_path), detectors.sample_fps),
        detector_factory(detectors),
        repo,
    )
    job = repo.claim_next()
    assert job is not None
    runner.run(job)


def test_full_flow(client: TestClient, repo: InMemoryJobRepository, tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "v.avi")
    with video.open("rb") as fh:
        video_id = client.post("/videos", files={"file": ("v.avi", fh)}).json()["video_id"]
    job = client.post("/jobs", json={"video_id": video_id}).json()
    assert job["status"] == "PENDING"
    run_one_job(repo)
    done = client.get(f"/jobs/{job['id']}").json()
    assert done["status"] == "DONE"
    assert done["segments"] == [{"start_s": 1.0, "end_s": 2.9}]


def test_validation_error_is_422(client: TestClient) -> None:
    assert client.post("/jobs", json={"detector": "x"}).status_code == 422


def test_unknown_video_is_404(client: TestClient) -> None:
    assert client.post("/jobs", json={"video_id": "nope"}).status_code == 404
