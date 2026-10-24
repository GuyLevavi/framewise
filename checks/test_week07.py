"""Week 7 acceptance: an image that is small, secret-free and runs as an arbitrary user.

The static checks always run. The build test runs only where an image can be built:
set FRAMEWISE_BUILD_IMAGE=1 and FRAMEWISE_BASE_IMAGE=<base image in Quay>.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from checks.fixtures import REPO, free_port

CONTAINERFILE = next(
    (REPO / n for n in ("Containerfile", "Dockerfile") if (REPO / n).exists()), None
)


def ignore_patterns() -> set[str]:
    for name in (".containerignore", ".dockerignore"):
        if (REPO / name).exists():
            lines = (REPO / name).read_text().splitlines()
            return {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    return set()


def test_build_is_multi_stage() -> None:
    assert CONTAINERFILE is not None, "add a Containerfile (or Dockerfile) at the repo root"
    stages = [ln for ln in CONTAINERFILE.read_text().splitlines() if ln.upper().startswith("FROM ")]
    assert len(stages) >= 2, "use a build stage and a runtime stage"


def test_runtime_stage_is_writable_by_group_zero() -> None:
    assert CONTAINERFILE is not None
    runtime = CONTAINERFILE.read_text().upper().split("\nFROM ")[-1]
    assert "CHGRP" in runtime and "G=U" in runtime, "OpenShift runs as a random UID in group 0"


def test_build_context_leaves_out_tests_configs_and_secrets() -> None:
    patterns = ignore_patterns()
    for needed in (".venv", ".git", "tests", ".env"):
        assert needed in patterns, f"{needed} should be in .containerignore"
    assert any("configs" in p for p in patterns), "configs are mounted at runtime, not baked in"


engine = shutil.which("podman") or shutil.which("docker")


@pytest.mark.skipif(
    not (
        engine
        and os.environ.get("FRAMEWISE_BUILD_IMAGE")
        and os.environ.get("FRAMEWISE_BASE_IMAGE")
    ),
    reason="set FRAMEWISE_BUILD_IMAGE=1 and FRAMEWISE_BASE_IMAGE, with podman or docker installed",
)
def test_image_builds_and_serves_as_a_random_uid(tmp_path: Path) -> None:
    assert engine is not None and CONTAINERFILE is not None
    tag = "framewise-check:latest"
    build = [engine, "build", "-f", str(CONTAINERFILE), "-t", tag]
    build += ["--build-arg", f"BASE_IMAGE={os.environ['FRAMEWISE_BASE_IMAGE']}", str(REPO)]
    subprocess.run(build, check=True, timeout=1800)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "api.yaml").write_text(
        "upload_dir: /app/data/uploads\ndetectors_config: /app/configs/detectors.yaml\n"
    )
    shutil.copy(REPO / "worker/configs/detectors.yaml", configs / "detectors.yaml")
    port = free_port()
    run = [engine, "run", "--rm", "-d", "--user", "1000620000:0", "-p", f"{port}:8000"]
    run += [
        "-v",
        f"{configs}:/app/configs:ro",
        "-e",
        "DATABASE_URL=postgresql+psycopg://x@nowhere/x",
    ]
    container = subprocess.run(
        [*run, tag], check=True, capture_output=True, text=True
    ).stdout.strip()
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                time.sleep(1)
        pytest.fail("container did not answer /health as UID 1000620000")
    finally:
        subprocess.run([engine, "rm", "-f", container], check=False)
