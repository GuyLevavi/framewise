from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.bump_image_tag import bump

REPO = Path(__file__).resolve().parent.parent


def test_bumps_only_image_tag_and_keeps_comments() -> None:
    text = "# header\nimage:\n  repository: q/fw\n  tag: old   # set by CI\nother:\n  tag: keep\n"
    out = bump(text, "3f2a9c1e")
    assert out == text.replace("tag: old", "tag: 3f2a9c1e")


def test_real_values_file_bumps_to_valid_yaml() -> None:
    out = bump((REPO / "deploy/chart/values-nonprod.yaml").read_text(), "abc1234")
    assert yaml.safe_load(out)["image"]["tag"] == "abc1234"


@pytest.mark.parametrize("bad", ["", "has space", "a" * 200, "x;rm -rf"])
def test_rejects_invalid_tags(bad: str) -> None:
    with pytest.raises(ValueError, match="valid image tag"):
        bump("image:\n  tag: x\n", bad)


def test_missing_tag_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"no image\.tag"):
        bump("image:\n  repository: x\nother: 1\n", "abc")
