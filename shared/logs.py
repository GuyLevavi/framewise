"""One JSON object per log line, so Splunk can search by field (job_id, level, event)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

# Fields that callers may pass with `extra={...}` and that end up in the JSON line.
CONTEXT_FIELDS = ("job_id", "video_id", "detector", "frames", "duration_s")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for name in CONTEXT_FIELDS:
            if name in record.__dict__:
                line[name] = record.__dict__[name]
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
