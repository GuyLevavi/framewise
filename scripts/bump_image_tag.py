"""Set `image.tag` in a Helm values file, keeping comments and layout. Used by the CI deploy step.

python scripts/bump_image_tag.py deploy/chart/values-nonprod.yaml 3f2a9c1e
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"^(?P<indent>\s+)tag:\s*\S+(?P<rest>.*)$")


def bump(text: str, tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError(f"not a valid image tag: {tag!r}")
    lines = text.splitlines(keepends=True)
    in_image = False
    for i, line in enumerate(lines):
        if re.match(r"^image:\s*$", line):
            in_image = True
            continue
        if in_image and line.strip() and not line[0].isspace():
            break  # left the image: block without finding a tag
        if in_image and (m := TAG_RE.match(line.rstrip("\n"))):
            lines[i] = f"{m['indent']}tag: {tag}{m['rest']}\n"
            return "".join(lines)
    raise ValueError("no image.tag found")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.write_text(bump(path.read_text(), sys.argv[2]))
