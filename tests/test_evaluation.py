from __future__ import annotations

import json
from pathlib import Path

from checks import synth
from shared.evaluation import ClassScore, LocalLabelSource, frames_from_label_studio, score_frames


def test_label_studio_ranges_become_zero_based_frames(tmp_path: Path) -> None:
    export = synth.write_label_studio_export(
        tmp_path / "a.json", tmp_path / "a.avi", "person", 10, 12
    )
    [task] = json.loads(export.read_text())
    assert frames_from_label_studio(task) == {10: {"person"}, 11: {"person"}, 12: {"person"}}


def test_scoring_hand_computed() -> None:
    truth = {1: {"person"}, 2: {"person"}, 3: {"person", "car"}}
    predicted = {0: {"car"}, 1: {"person"}, 2: set(), 3: {"person", "car"}, 4: set()}
    person, car = score_frames(predicted, truth, ["person", "car"])
    assert person == ClassScore("person", true_pos=2, false_pos=0, false_neg=1)
    assert (person.precision, round(person.recall, 3)) == (1.0, 0.667)
    assert car == ClassScore("car", true_pos=1, false_pos=1, false_neg=0)
    assert car.precision == 0.5


def test_nothing_to_find_and_nothing_found_scores_perfectly() -> None:
    [s] = score_frames({0: set()}, {}, ["person"])
    assert (s.precision, s.recall) == (1.0, 1.0)


def test_local_source_pairs_exports_with_videos_and_versions_the_labels(tmp_path: Path) -> None:
    synth.write_label_studio_export(tmp_path / "clip.json", tmp_path / "clip.avi", "person", 0, 1)
    source = LocalLabelSource(tmp_path)
    [clip] = list(source.clips())
    assert clip.video_path == tmp_path / "clip.avi" and clip.frames == {
        0: {"person"},
        1: {"person"},
    }
    before = source.version()
    synth.write_label_studio_export(tmp_path / "clip.json", tmp_path / "clip.avi", "person", 0, 2)
    assert source.version() != before  # relabeling changes the data version
