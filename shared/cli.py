"""Command-line entry point: `framewise <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="framewise")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("probe", help="print video metadata as JSON")
    p.add_argument("video", type=Path)
    p.add_argument("--save-every", type=int, default=0, metavar="N", help="save every N-th frame")
    p.add_argument("--out", type=Path, default=Path("out/frames"))
    p.set_defaults(func=_cmd_probe)
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
    except VideoError as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
