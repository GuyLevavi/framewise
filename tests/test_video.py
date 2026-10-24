from __future__ import annotations

from pathlib import Path

import pytest

from checks import synth
from shared.video import VideoError, iter_frames, probe


def test_probe_reads_synthetic_video(tmp_path: Path) -> None:
    info = probe(synth.write_moving_square(tmp_path / "v.avi"))
    assert (info.width, info.height, info.frame_count) == (synth.WIDTH, synth.HEIGHT, synth.FRAMES)


def test_iter_frames_samples_every_n(tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "v.avi")
    indices = [i for i, _ in iter_frames(video, every_n=7)]
    assert indices == [0, 7, 14, 21, 28, 35]


def test_iter_frames_rejects_bad_step(tmp_path: Path) -> None:
    video = synth.write_moving_square(tmp_path / "v.avi")
    with pytest.raises(ValueError, match="every_n"):
        list(iter_frames(video, every_n=0))


def test_corrupt_video_raises_video_error(tmp_path: Path) -> None:
    with pytest.raises(VideoError):
        probe(synth.write_corrupt(tmp_path / "bad.mp4"))
