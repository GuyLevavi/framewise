"""Week 3 acceptance: the HTTP API, used only over HTTP."""

from __future__ import annotations

from pathlib import Path

import httpx

from checks import synth
from checks.conftest import poll_job


def upload(url: str, video: Path) -> str:
    with video.open("rb") as fh:
        r = httpx.post(f"{url}/videos", files={"file": (video.name, fh, "video/x-msvideo")})
    assert r.status_code == 201, r.text
    return str(r.json()["video_id"])


def test_health(api_url: str) -> None:
    assert httpx.get(f"{api_url}/health").status_code == 200


def test_upload_submit_poll_gives_the_motion_segment(api_url: str, tmp_path: Path) -> None:
    video_id = upload(api_url, synth.write_moving_square(tmp_path / "v.avi"))
    r = httpx.post(f"{api_url}/jobs", json={"video_id": video_id, "detector": "motion-sensitive"})
    assert r.status_code == 202, r.text
    job = poll_job(api_url, str(r.json()["id"]))
    assert job["status"] == "DONE", job
    segments = job["segments"]
    assert isinstance(segments, list) and len(segments) == 1


def test_unknown_job_is_404_json(api_url: str) -> None:
    r = httpx.get(f"{api_url}/jobs/does-not-exist")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_unknown_video_is_404(api_url: str) -> None:
    r = httpx.post(f"{api_url}/jobs", json={"video_id": "nope"})
    assert r.status_code == 404


def test_unknown_detector_is_422(api_url: str, tmp_path: Path) -> None:
    video_id = upload(api_url, synth.write_still(tmp_path / "s.avi"))
    r = httpx.post(f"{api_url}/jobs", json={"video_id": video_id, "detector": "no-such"})
    assert r.status_code == 422


def test_corrupt_video_fails_with_reason(api_url: str, tmp_path: Path) -> None:
    video_id = upload(api_url, synth.write_corrupt(tmp_path / "bad.avi"))
    r = httpx.post(f"{api_url}/jobs", json={"video_id": video_id})
    job = poll_job(api_url, str(r.json()["id"]))
    assert job["status"] == "FAILED"
    assert job["error"]


def test_openapi_documents_every_endpoint(api_url: str) -> None:
    paths = httpx.get(f"{api_url}/openapi.json").json()["paths"]
    for path in ("/health", "/videos", "/jobs", "/jobs/{job_id}"):
        assert path in paths
