"""Week 2 acceptance: `framewise analyze` finds the known motion window."""

from __future__ import annotations

import json
from pathlib import Path

from checks import synth
from checks.test_week01 import run_cli

ONE_FRAME = 1 / synth.FPS


def analyze(video: Path, *extra: str) -> dict[str, list[dict[str, float]]]:
    result = run_cli("analyze", str(video), *extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_moving_square_gives_one_segment_within_one_frame(tmp_path: Path) -> None:
    out = analyze(synth.write_moving_square(tmp_path / "v.avi"))
    assert len(out["segments"]) == 1
    seg = out["segments"][0]
    assert abs(seg["start_s"] - synth.MOTION_START / synth.FPS) <= ONE_FRAME + 1e-6
    assert abs(seg["end_s"] - synth.MOTION_END / synth.FPS) <= ONE_FRAME + 1e-6


def test_still_video_gives_no_segments(tmp_path: Path) -> None:
    assert analyze(synth.write_still(tmp_path / "still.avi"))["segments"] == []


def test_new_variant_needs_only_yaml(tmp_path: Path) -> None:
    config = tmp_path / "detectors.yaml"
    config.write_text(
        "sample_fps: 10\nmax_gap_s: 0.5\n"
        "detectors:\n  my-variant:\n    kind: motion\n    threshold: 20\n    min_area: 30\n"
    )
    video = synth.write_moving_square(tmp_path / "v.avi")
    out = analyze(video, "--detector", "my-variant", "--config", str(config))
    assert len(out["segments"]) == 1


def test_bad_config_fails_cleanly(tmp_path: Path) -> None:
    config = tmp_path / "detectors.yaml"
    config.write_text("detectors:\n  x:\n    kind: motion\n    threshold: -5\n")
    result = run_cli("analyze", str(synth.write_still(tmp_path / "s.avi")), "--config", str(config))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "threshold" in result.stderr, "name the setting that is wrong"
