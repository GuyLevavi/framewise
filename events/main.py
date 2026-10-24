"""Events API: composite events over the detections the worker stored. Read-only."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from shared.auth import require_api_key, warn_if_open
from shared.config import load_yaml
from shared.db import PostgresJobRepository, make_engine
from shared.events import RulesFile, find_episodes
from shared.jobs import NotFoundError

CONFIG_ENV = "FRAMEWISE_EVENTS_CONFIG"


class EpisodeOut(BaseModel):
    key: str
    rule: str
    start_s: float
    end_s: float
    frames: int


def create_app(rules: RulesFile | None = None, engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = Path(os.environ.get(CONFIG_ENV, "events/configs/events.yaml"))
        app.state.rules = rules or load_yaml(path, RulesFile)
        app.state.engine = engine or make_engine()
        warn_if_open()
        yield

    app = FastAPI(title="Framewise events", lifespan=lifespan)
    protected = APIRouter(dependencies=[Depends(require_api_key)])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @protected.get("/videos/{video_id}/events")
    def video_events(video_id: str, request: Request, rule: str | None = None) -> list[EpisodeOut]:
        rules_file: RulesFile = request.app.state.rules
        db: Engine = request.app.state.engine
        try:
            PostgresJobRepository(db).get_video(video_id)
        except NotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown video {video_id}") from None
        if rule is not None and rule not in rules_file.rules:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown rule {rule}")
        names = [rule] if rule else sorted(rules_file.rules)
        return [
            EpisodeOut(key=e.key, rule=e.rule, start_s=e.start_s, end_s=e.end_s, frames=e.frames)
            for name in names
            for e in find_episodes(db, video_id, name, rules_file.rules[name])
        ]

    app.include_router(protected)
    return app


app = create_app()
