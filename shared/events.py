"""Composite events from stored detections, computed in SQL.

Example rule: "a person box overlaps a car box". Overlapping frames are grouped into episodes:
consecutive frames closer than `max_gap_s` belong together (the "gaps and islands" pattern).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    a: str  # label of the first object, e.g. person
    b: list[str]  # labels it must overlap, e.g. [car, truck]
    min_overlap: float = Field(default=0.1, gt=0, le=1)  # intersection / smaller box area
    min_duration_s: float = Field(default=0.0, ge=0)
    max_gap_s: float = Field(default=0.5, ge=0)


class RulesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rules: dict[str, Rule]


@dataclass(frozen=True)
class Episode:
    key: str
    rule: str
    start_s: float
    end_s: float
    frames: int


EPISODES_SQL = text(
    """
    WITH latest_job AS (
        SELECT id FROM jobs
        WHERE video_id = :video_id AND status = 'DONE'
        ORDER BY created_at DESC LIMIT 1
    ),
    hits AS (                                   -- frames where an `a` box overlaps a `b` box
        SELECT DISTINCT a.frame_time
        FROM detections a
        JOIN detections b
          ON b.job_id = a.job_id AND b.frame_index = a.frame_index AND b.id <> a.id
        WHERE a.job_id = (SELECT id FROM latest_job)
          AND a.label = :a AND b.label IN :b
          AND GREATEST(0, LEAST(a.x2, b.x2) - GREATEST(a.x1, b.x1))
            * GREATEST(0, LEAST(a.y2, b.y2) - GREATEST(a.y1, b.y1))
            >= :min_overlap * LEAST((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1))
    ),
    marked AS (                                 -- 1 where a new episode starts
        SELECT frame_time,
               CASE WHEN frame_time - LAG(frame_time) OVER (ORDER BY frame_time) <= :max_gap_s
                    THEN 0 ELSE 1 END AS starts
        FROM hits
    ),
    islands AS (
        SELECT frame_time, SUM(starts) OVER (ORDER BY frame_time) AS episode FROM marked
    )
    SELECT MIN(frame_time) AS start_s, MAX(frame_time) AS end_s, COUNT(*) AS frames
    FROM islands
    GROUP BY episode
    HAVING MAX(frame_time) - MIN(frame_time) >= :min_duration_s
    ORDER BY start_s
    """
).bindparams(bindparam("b", expanding=True))


def episode_key(video_id: str, rule: str, start_s: float) -> str:
    """Same video, rule and start time give the same key, so reruns never duplicate an episode."""
    return hashlib.sha256(f"{video_id}|{rule}|{start_s:.3f}".encode()).hexdigest()[:16]


def find_episodes(engine: Engine, video_id: str, name: str, rule: Rule) -> list[Episode]:
    params = {"video_id": video_id, **rule.model_dump()}
    with engine.connect() as conn:
        rows = conn.execute(EPISODES_SQL, params).all()
    return [
        Episode(episode_key(video_id, name, float(r.start_s)), name, r.start_s, r.end_s, r.frames)
        for r in rows
    ]
