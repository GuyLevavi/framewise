"""Fixtures that run the app the way a user would: as a real process.

Contract the trainee's code must keep (see TRAINEE.md):
- the API is importable as `api.main:app`
- it reads its YAML config from the path in FRAMEWISE_API_CONFIG
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parent.parent


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


def write_api_config(tmp: Path, extra: str = "") -> Path:
    config = tmp / "api.yaml"
    config.write_text(
        f"upload_dir: {tmp / 'uploads'}\n"
        f"detectors_config: {REPO / 'worker/configs/detectors.yaml'}\n" + extra
    )
    return config


@pytest.fixture
def api_env(tmp_path: Path) -> dict[str, str]:
    """Environment for the API process; later weeks add database settings here."""
    return {"FRAMEWISE_API_CONFIG": str(write_api_config(tmp_path))}


@pytest.fixture
def api_url(api_env: dict[str, str]) -> Iterator[str]:
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)],
        cwd=REPO,
        env={**os.environ, **api_env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        wait_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def poll_job(url: str, job_id: str, timeout_s: float = 60) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body: dict[str, object] = httpx.get(f"{url}/jobs/{job_id}", timeout=5).json()
        if body["status"] in ("DONE", "FAILED"):
            return body
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")
