"""Week 10 acceptance: API key, upload limits and the events API, over HTTP.

Contract: the events API is importable as `events.main:app`; both APIs read FRAMEWISE_API_KEY.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from checks import synth
from checks.fixtures import (
    free_port,
    poll_job,
    start,
    stop,
    wait_healthy,
    write_api_config,
    write_worker_config,
)

KEY = {"X-API-Key": "check-key"}


@pytest.fixture
def app_env(tmp_path: Path, clean_db: str) -> dict[str, str]:
    return {
        "FRAMEWISE_API_CONFIG": str(write_api_config(tmp_path, "max_upload_mb: 1\n")),
        "FRAMEWISE_WORKER_CONFIG": str(write_worker_config(tmp_path)),
        "DATABASE_URL": clean_db,
        "FRAMEWISE_API_KEY": KEY["X-API-Key"],
    }


@pytest.fixture
def events_url(app_env: dict[str, str], tmp_path: Path) -> Iterator[str]:
    port = free_port()
    proc = start(
        ["-m", "uvicorn", "events.main:app", "--port", str(port)], app_env, tmp_path / "ev.log"
    )
    url = f"http://127.0.0.1:{port}"
    try:
        wait_healthy(url, proc)
        yield url
    finally:
        stop(proc)


def test_endpoints_need_the_key_but_health_does_not(api_url: str) -> None:
    assert httpx.get(f"{api_url}/health").status_code == 200
    assert httpx.get(f"{api_url}/jobs/x").status_code == 401
    assert httpx.get(f"{api_url}/jobs/x", headers={"X-API-Key": "wrong"}).status_code == 401
    assert httpx.get(f"{api_url}/jobs/x", headers=KEY).status_code == 404


def test_upload_limits(api_url: str) -> None:
    files = {"file": ("notes.txt", b"hello")}
    assert httpx.post(f"{api_url}/videos", files=files, headers=KEY).status_code == 415
    big = {"file": ("big.mp4", b"x" * (1024 * 1024 + 1))}
    assert httpx.post(f"{api_url}/videos", files=big, headers=KEY).status_code == 413


def test_events_api_answers_for_a_processed_video(
    api_url: str, events_url: str, tmp_path: Path
) -> None:
    video = synth.write_still(tmp_path / "still.avi")
    with video.open("rb") as fh:
        video_id = httpx.post(
            f"{api_url}/videos", files={"file": (video.name, fh)}, headers=KEY
        ).json()["video_id"]
    job_id = httpx.post(f"{api_url}/jobs", json={"video_id": video_id}, headers=KEY).json()["id"]
    assert poll_job(api_url, job_id, headers=KEY)["status"] == "DONE"
    r = httpx.get(f"{events_url}/videos/{video_id}/events", headers=KEY)
    assert r.status_code == 200 and r.json() == []
    assert httpx.get(f"{events_url}/videos/nope/events", headers=KEY).status_code == 404
    assert httpx.get(f"{events_url}/videos/{video_id}/events").status_code == 401
