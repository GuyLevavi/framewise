"""Command-line entry point: `framewise <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from shared.config import DetectorsFile, detector_factory, load_yaml
from shared.jobs import InMemoryJobRepository, Job, JobStatus
from shared.pipeline import PipelineRunner, VideoFileSource, group_segments
from shared.video import VideoError, probe, save_frames

log = logging.getLogger("framewise")


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.video)
    result: dict[str, object] = info.to_dict()
    if args.save_every:
        saved = save_frames(args.video, args.out, args.save_every)
        result["saved_frames"] = len(saved)
        log.info("saved %d frames to %s", len(saved), args.out)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    config = load_yaml(args.config, DetectorsFile)
    repository = InMemoryJobRepository()
    runner = PipelineRunner(
        sources=lambda job: VideoFileSource(Path(job.video_path), config.sample_fps),
        detectors=detector_factory(config),
        repository=repository,
    )
    job = Job(video_id="cli", video_path=str(args.video), detector=args.detector)
    repository.add(job)
    runner.run(job)
    finished = repository.get(job.id)
    if finished.status is JobStatus.FAILED:
        log.error("analysis failed: %s", finished.error)
        return 2
    times = [f.time_s for f in repository.list_detections(job.id) if f.detections]
    segments = group_segments(times, config.max_gap_s)
    print(json.dumps({"segments": [vars(s) for s in segments]}, indent=2))
    return 0


def _cmd_worker(_args: argparse.Namespace) -> int:
    from worker.main import main as worker_main  # imported lazily: needs the database packages

    worker_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="framewise")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("probe", help="print video metadata as JSON")
    p.add_argument("video", type=Path)
    p.add_argument("--save-every", type=int, default=0, metavar="N", help="save every N-th frame")
    p.add_argument("--out", type=Path, default=Path("out/frames"))
    p.set_defaults(func=_cmd_probe)

    a = commands.add_parser("analyze", help="detect motion segments in a video")
    a.add_argument("video", type=Path)
    a.add_argument("--detector", default="motion-sensitive")
    a.add_argument("--config", type=Path, default=Path("worker/configs/detectors.yaml"))
    a.set_defaults(func=_cmd_analyze)

    w = commands.add_parser("worker", help="run the job worker (needs DATABASE_URL)")
    w.set_defaults(func=_cmd_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.func(args))
    except (VideoError, OSError, ValidationError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
