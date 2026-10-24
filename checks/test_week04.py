"""Week 4 acceptance: jobs persist in Postgres and run in a separate worker."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text

from checks import synth
from checks.fixtures import REPO, free_port, poll_job, start, stop, wait_healthy
from checks.test_week03 import upload


def submit(url: str, video_id: str) -> str:
    r = httpx.post(f"{url}/jobs", json={"video_id": video_id})
    assert r.status_code == 202, r.text
    return str(r.json()["id"])


def test_schema_builds_from_empty_database(clean_db: str) -> None:
    engine = create_engine(clean_db)
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            ).scalars()
        )
        indexed = set(
            conn.execute(
                text(
                    "SELECT tablename FROM pg_indexes"
                    " WHERE schemaname='public' AND indexname NOT LIKE '%pkey'"
                )
            ).scalars()
        )
    engine.dispose()
    assert {"videos", "jobs", "detections"} <= tables
    assert {"jobs", "detections"} <= indexed, (
        "jobs and detections each need an index for how they are read"
    )


@pytest.mark.parametrize("worker_count", [3])
def test_parallel_jobs_finish_while_api_stays_responsive(api_url: str, tmp_path: Path) -> None:
    video_id = upload(api_url, synth.write_moving_square(tmp_path / "v.avi", frames=200))
    job_ids = [submit(api_url, video_id) for _ in range(3)]
    for _ in range(10):
        started = time.monotonic()
        assert httpx.get(f"{api_url}/health", timeout=2).status_code == 200
        assert time.monotonic() - started < 0.5, "/health is slow: is work running in the API?"
        time.sleep(0.1)
    for job_id in job_ids:
        assert poll_job(api_url, job_id)["status"] == "DONE"


def test_results_survive_an_api_restart(app_env: dict[str, str], tmp_path: Path) -> None:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    api_cmd = ["-m", "uvicorn", "api.main:app", "--port", str(port)]
    worker = start(["-m", "worker"], app_env, tmp_path / "worker.log")
    api = start(api_cmd, app_env, tmp_path / "api1.log")
    try:
        wait_healthy(url, api)
        job_id = submit(url, upload(url, synth.write_moving_square(tmp_path / "v.avi")))
        before = poll_job(url, job_id)
        stop(api)
        api = start(api_cmd, app_env, tmp_path / "api2.log")
        wait_healthy(url, api)
        after = httpx.get(f"{url}/jobs/{job_id}").json()
        assert after == before
    finally:
        stop(api)
        stop(worker)


def test_migrations_downgrade_and_upgrade_cleanly(clean_db: str) -> None:
    env = {"DATABASE_URL": clean_db}
    for step in (["downgrade", "base"], ["upgrade", "head"]):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *step],
            cwd=REPO,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
