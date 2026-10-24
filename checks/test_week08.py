"""Week 8 acceptance (static part): the cluster manifests have what OpenShift needs.

The live part, done with the tutor in the cluster: the Route answers, `oc delete pod`
recovers, three workers share jobs, and one Splunk search on a job_id shows the whole job.
"""

from __future__ import annotations

from checks.deploy_files import deploy_text
from checks.fixtures import REPO


def test_manifests_have_probes_config_route_and_shared_storage() -> None:
    text = deploy_text()
    for needed in (
        "readinessProbe",
        "livenessProbe",
        "kind: ConfigMap",
        "kind: Route",
        "ReadWriteMany",
    ):
        assert needed in text, f"no {needed} in deploy/"


def test_database_url_comes_from_a_secret() -> None:
    files = [f for f in (REPO / "deploy/chart").rglob("*") if f.suffix in (".yaml", ".tpl")]
    cluster = "\n".join(f.read_text() for f in files)
    assert "secretKeyRef" in cluster, "DATABASE_URL should come from a Secret"
    assert "postgresql" not in cluster, "no database URL or password in the cluster manifests"
