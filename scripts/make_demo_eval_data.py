"""Write one labeled demo clip into eval/data, so `pytest -m eval` runs on a fresh clone.

The clip shows the astronaut test image (one person) on frames 10 to 29, with a matching
Label Studio export. Real evaluation data comes from DagsHub and Label Studio through the
team helper library; this is only for trying the gate locally.

    uv run python scripts/make_demo_eval_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from checks import synth  # noqa: E402  (needs the repo on sys.path first)


def main() -> None:
    data = REPO / "eval/data"
    data.mkdir(parents=True, exist_ok=True)
    video = synth.write_image_clip(
        data / "astronaut.avi", REPO / "checks/data/astronaut.jpg", start=10, end=29
    )
    synth.write_label_studio_export(data / "astronaut.json", video, "person", 10, 29)
    print(f"wrote {video.name} and astronaut.json to {data}")


if __name__ == "__main__":
    main()
