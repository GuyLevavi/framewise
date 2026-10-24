"""Week 9 acceptance (static part): a Helm chart Argo CD can sync, migrations as a PreSync hook.

`helm lint` and `helm template` run when helm is installed. The live part, with the tutor:
a merge to main reaches nonprod with no manual step, and `git revert` rolls it back.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import yaml

from checks.deploy_files import deploy_text
from checks.fixtures import REPO

CHART = REPO / "deploy/chart"


def test_chart_and_nonprod_values_exist() -> None:
    assert (CHART / "Chart.yaml").exists()
    values = yaml.safe_load((CHART / "values-nonprod.yaml").read_text())
    assert "tag" in values["image"], "CI bumps image.tag in values-nonprod.yaml"


def test_migrations_run_as_an_argo_presync_hook() -> None:
    text = deploy_text()
    assert "argocd.argoproj.io/hook: PreSync" in text
    assert "alembic" in text and "upgrade" in text


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed here")
def test_helm_lint_and_template() -> None:
    subprocess.run(["helm", "lint", str(CHART)], check=True)
    out = subprocess.run(
        ["helm", "template", "fw", str(CHART), "-f", str(CHART / "values-nonprod.yaml")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    kinds = {doc["kind"] for doc in yaml.safe_load_all(out) if doc}
    assert {"Deployment", "Service", "Route", "ConfigMap", "Job"} <= kinds
