"""Week 1 acceptance: the `framewise probe` command, used only from the outside."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from checks import synth


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "shared.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_probe_prints_metadata_as_json(tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "square.avi")
    result = run_cli("probe", str(video))
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["width"] == synth.WIDTH
    assert info["height"] == synth.HEIGHT
    assert info["frame_count"] == synth.FRAMES
    assert abs(info["fps"] - synth.FPS) < 0.01
    assert abs(info["duration_s"] - synth.FRAMES / synth.FPS) < 0.01


def test_probe_saves_every_nth_frame(tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "square.avi")
    out = tmp_path / "frames"
    result = run_cli("probe", str(video), "--save-every", "10", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("*.jpg"))) == synth.FRAMES // 10


def test_corrupt_file_gives_clear_error_not_traceback(tmp_path: Path) -> None:
    bad = synth.write_corrupt(tmp_path / "bad.mp4")
    result = run_cli("probe", str(bad))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "bad.mp4" in result.stderr


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    result = run_cli("probe", str(tmp_path / "nope.mp4"))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
