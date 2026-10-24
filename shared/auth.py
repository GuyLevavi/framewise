"""API key check: the client sends a shared secret in a header; the value lives in a Secret.

If FRAMEWISE_API_KEY is not set the check is off (local development); the cluster always sets it.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

API_KEY_ENV = "FRAMEWISE_API_KEY"
log = logging.getLogger(__name__)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong X-API-Key")


def warn_if_open() -> None:
    if not os.environ.get(API_KEY_ENV):
        log.warning(
            "%s is not set: endpoints are open (fine locally, never in the cluster)", API_KEY_ENV
        )
