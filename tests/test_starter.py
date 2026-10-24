"""The starter must hold no reference code, and its checks must fail until the trainee works."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.make_starter import make_starter

REPO = Path(__file__).resolve().parent.parent
REFERENCE_ONLY = (
    "shared/db.py",
    "shared/pipeline.py",
    "api/main.py",
    "worker/main.py",
    "events/main.py",
    "eval/test_gate.py",
    "tests/test_db.py",
    "migrations",
    "Containerfile",
    "deploy",
    ".gitlab-ci.yml",
    "TUTOR.md",
)


def test_starter_has_checks_and_configs_but_no_reference_code(tmp_path: Path) -> None:
    make_starter(tmp_path)
    for needed in ("checks/test_week01.py", "checks/fixtures.py", "pyproject.toml", "TRAINEE.md"):
        assert (tmp_path / needed).exists(), needed
    assert (tmp_path / "worker/configs/detectors.yaml").exists()
    for secret in REFERENCE_ONLY:
        assert not (tmp_path / secret).exists(), f"{secret} leaked into the starter"


FAST_CHECKS = [f"checks/test_week{n:02d}.py" for n in (1, 2, 7, 8, 9)]  # no database needed


def run_checks(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *FAST_CHECKS],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_no_check_passes_on_the_empty_starter(tmp_path: Path) -> None:
    make_starter(tmp_path)
    result = run_checks(tmp_path)
    assert result.returncode == 1, result.stdout  # 1 = tests ran and failed, not a crash
    summary = result.stdout.strip().splitlines()[-1]
    assert "failed" in summary and "passed" not in summary, f"a check passes vacuously: {summary}"


def test_reference_passes_the_same_checks() -> None:
    assert run_checks(REPO).returncode == 0
