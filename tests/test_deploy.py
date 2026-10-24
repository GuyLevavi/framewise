"""Deployment files must stay in step with the code's settings models."""

from __future__ import annotations

from pathlib import Path

import yaml

from api.settings import ApiSettings
from shared.config import DetectorsFile
from worker.main import WorkerSettings

REPO = Path(__file__).resolve().parent.parent


def test_local_pod_configs_validate() -> None:
    folder = REPO / "deploy/local-configs"
    ApiSettings.model_validate(yaml.safe_load((folder / "api.yaml").read_text()))
    WorkerSettings.model_validate(yaml.safe_load((folder / "worker.yaml").read_text()))
    DetectorsFile.model_validate(yaml.safe_load((folder / "detectors.yaml").read_text()))


def test_helm_values_configs_validate() -> None:
    values = yaml.safe_load((REPO / "deploy/chart/values.yaml").read_text())["config"]
    ApiSettings.model_validate(values["api"])
    WorkerSettings.model_validate(values["worker"])
    DetectorsFile.model_validate(values["detectors"])


def test_pod_runs_api_worker_and_postgres() -> None:
    pod = yaml.safe_load((REPO / "deploy/pod.yaml").read_text())
    assert {c["name"] for c in pod["spec"]["containers"]} == {"postgres", "api", "worker"}
