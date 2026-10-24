"""Fixtures that run the app the way a user would: as real processes against a real database.

Contract the trainee's code must keep (see TRAINEE.md):
- the API is importable as `api.main:app` and reads its YAML config from FRAMEWISE_API_CONFIG
- from week 4: `python -m worker` runs the worker, configured by FRAMEWISE_WORKER_CONFIG
- from week 4: both read DATABASE_URL, and `alembic upgrade head` builds the schema
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

REPO = Path(__file__).resolve().parent.parent
ADMIN_URL_ENV = "FRAMEWISE_TEST_DATABASE_URL"
DEFAULT_ADMIN_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
HAS_WORKER = (REPO / "worker" / "main.py").exists()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_healthy(url: str, proc: subprocess.Popen[bytes], timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"{url}/health did not answer within {timeout_s}s")


# ---------------------------------------------------------------- database


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A fresh database for this test session, migrated with `alembic upgrade head`."""
    admin_url = os.environ.get(ADMIN_URL_ENV, DEFAULT_ADMIN_URL)
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect():
            pass
    except Exception as exc:
        pytest.fail(
            f"Postgres is not reachable at {make_url(admin_url).render_as_string()}: {exc}\n"
            f"Start it, or point {ADMIN_URL_ENV} at a server where you may create databases."
        )
    name = f"framewise_check_{uuid.uuid4().hex[:8]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(admin_url).set(database=name).render_as_string(hide_password=False)
    migrate = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )
    assert migrate.returncode == 0, f"alembic upgrade head failed:\n{migrate.stderr}"
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def clean_db(database_url: str) -> str:
    """The session database, emptied before the test."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        ).scalars()
        names = [t for t in tables if t != "alembic_version"]
        if names:
            conn.execute(text(f"TRUNCATE {', '.join(names)} CASCADE"))
    engine.dispose()
    return database_url


# ---------------------------------------------------------------- running the app


def write_api_config(tmp: Path, extra: str = "") -> Path:
    config = tmp / "api.yaml"
    config.write_text(
        f"upload_dir: {tmp / 'uploads'}\n"
        f"detectors_config: {REPO / 'worker/configs/detectors.yaml'}\n" + extra
    )
    return config


def write_worker_config(tmp: Path) -> Path:
    config = tmp / "worker.yaml"
    config.write_text(
        f"detectors_config: {REPO / 'worker/configs/detectors.yaml'}\n"
        "poll_interval_s: 0.2\nstale_after_s: 60\nmax_attempts: 3\n"
    )
    return config


@pytest.fixture
def app_env(tmp_path: Path, request: pytest.FixtureRequest) -> dict[str, str]:
    env = {
        "FRAMEWISE_API_CONFIG": str(write_api_config(tmp_path)),
        "FRAMEWISE_WORKER_CONFIG": str(write_worker_config(tmp_path)),
    }
    if HAS_WORKER:
        env["DATABASE_URL"] = request.getfixturevalue("clean_db")
    return env


def start(args: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as out:
        return subprocess.Popen(
            [sys.executable, *args], cwd=REPO, env={**os.environ, **env}, stdout=out, stderr=out
        )


def stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def worker_count() -> int:
    return 1


@pytest.fixture
def api_url(app_env: dict[str, str], tmp_path: Path, worker_count: int) -> Iterator[str]:
    """Base URL of a running API; from week 4 also starts `worker_count` worker processes."""
    port = free_port()
    procs = [
        start(["-m", "uvicorn", "api.main:app", "--port", str(port)], app_env, tmp_path / "api.log")
    ]
    if HAS_WORKER:
        procs += [
            start(["-m", "worker"], app_env, tmp_path / f"worker{i}.log")
            for i in range(worker_count)
        ]
    url = f"http://127.0.0.1:{port}"
    try:
        try:
            wait_healthy(url, procs[0])
        except (RuntimeError, TimeoutError) as exc:
            tail = "\n".join((tmp_path / "api.log").read_text().splitlines()[-15:])
            raise RuntimeError(f"{exc}\n--- last lines of the API log ---\n{tail}") from None
        yield url
    finally:
        for proc in procs:
            stop(proc)


def poll_job(
    url: str, job_id: str, timeout_s: float = 60, headers: dict[str, str] | None = None
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = httpx.get(f"{url}/jobs/{job_id}", timeout=5, headers=headers)
        body: dict[str, object] = r.json()
        if body["status"] in ("DONE", "FAILED"):
            return body
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")
