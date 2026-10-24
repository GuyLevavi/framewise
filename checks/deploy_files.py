"""Helpers for the static deployment checks (weeks 8 and 9)."""

from __future__ import annotations

from checks.fixtures import REPO


def deploy_text() -> str:
    """All YAML and template text under deploy/, joined, for simple presence checks."""
    files = sorted((REPO / "deploy").rglob("*.yaml")) + sorted((REPO / "deploy").rglob("*.tpl"))
    return "\n".join(f.read_text() for f in files if "local-configs" not in f.parts)
