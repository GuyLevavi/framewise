"""API settings from a YAML file; the path can be overridden for tests and deployments."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shared.config import load_yaml

CONFIG_ENV = "FRAMEWISE_API_CONFIG"
DEFAULT_CONFIG = Path("api/configs/api.yaml")


class ApiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    upload_dir: Path
    max_upload_mb: int = Field(default=200, gt=0)
    detectors_config: Path
    allowed_extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv"]


def load_settings() -> ApiSettings:
    return load_yaml(Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)), ApiSettings)
