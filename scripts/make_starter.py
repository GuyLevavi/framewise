"""Build the trainee's starting repo from this reference repo.

The trainee gets the acceptance checks, configs, tooling and empty packages. The reference
implementation, its unit tests and the deliverables they must write (CI, Containerfile, chart,
migrations, evaluation gate) stay with the tutor.

    python scripts/make_starter.py ../framewise-starter
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tracked files the trainee receives as they are.
KEEP = (
    ".gitignore",
    ".python-version",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "uv.toml.example",
    "conftest.py",
    "TRAINEE.md",
    "checks/",
    "api/configs/",
    "worker/configs/",
    "events/configs/",
    "eval/eval.yaml",
    "eval/data/.gitignore",
    "model_repository/",
    "scripts/make_demo_eval_data.py",
    ".gitlab/",
)
# Empty packages, so the project installs and imports resolve from day one.
PACKAGES = ("shared", "api", "worker", "events", "eval", "tests", "scripts")

README = """# Framewise (your repo)

Start with TRAINEE.md. Each week, make that week's checks pass:

```bash
uv sync
uv run pytest checks/test_week01.py
```
"""


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return out.splitlines()


def make_starter(target: Path) -> list[str]:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    copied: list[str] = []
    for name in tracked_files():
        if any(name == k or (k.endswith("/") and name.startswith(k)) for k in KEEP):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target / name)
            copied.append(name)
    for package in PACKAGES:
        (target / package).mkdir(parents=True, exist_ok=True)
        (target / package / "__init__.py").touch()
    (target / "README.md").write_text(README)
    return copied


if __name__ == "__main__":
    files = make_starter(Path(sys.argv[1]))
    print(f"starter written to {sys.argv[1]}: {len(files)} files plus empty packages")
