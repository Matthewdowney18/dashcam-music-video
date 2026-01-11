from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .ingest import DashcamConfig, discover_pairs, print_ingest_summary
from .events import detect_motion_events, MotionEvent
from .audio_events import detect_audio_events_for_clip
from .layout import make_vertical_test_output


def _get_pair(base_dir: Path, index: int):
    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)
    if not pairs:
        raise RuntimeError("No clip pairs found.")
    if index < 0 or index >= len(pairs):
        raise IndexError(f"--index {index} out of range (found {len(pairs)})")
    return pairs[index]


def _fmt_events(events: List[object]) -> None:
    if not events:
        print("  (none)")
    for ev in events:
        print(" ", ev)


# -----------------------------
# Commands
# -----------------------------

def cmd_ingest(args):
    cfg = DashcamConfig(
        base_dir=Path(args.base_dir),
        road_dir_name=args.road_dir,
        cabin_dir_name=args.cabin_dir,
    )
    pairs = discover_pairs(cfg)
    print_ingest_summary(pairs, cfg, show=args.show)


def cmd_motion(args):
    pair = _get_pair(Path(args.base_dir), args.index)
    cams = ["road", "cabin"] if args.camera == "both" else [args.camera]

    for cam in cams:
        path = pair.road if cam == "road" else pair.cabin
        print(f"\n[motion] {cam.upper()} — {path.name}")

        events = detect_motion_events(
            video_path=path,
            downscale_width=args.downscale_width,
            frame_step=args.frame_step,
            pre_event_sec=args.pre_event_sec,
            post_event_sec=args.post_event_sec,
            min_event_gap_sec=args.min_event_gap_sec,
            threshold_std=args.threshold_std,
            max_events=args.max_events,
        )
        _fmt_events(events)


def cmd_audio(args):
    pair = _get_pair(Path(args.base_dir), args.index)
    cams = ["road", "cabin"] if args.camera == "both" else [args.camera]

    for cam in cams:
        path = pair.road if cam == "road" else pair.cabin
        print(f"\n[audio] {cam.upper()} — {path.name}")

        events, features = detect_audio_events_for_clip(
            video_path=path,
            sample_rate=args.sample_rate,
            k=args.k,
            pre_event_sec=args.pre_event_sec,
            post_event_sec=args.post_event_sec,
            min_event_gap_sec=args.min_event_gap_sec,
            max_events=args.max_events,
        )
        _fmt_events(events)

        if args.debug_npy_prefix:
            import numpy as np
            prefix = f"{args.debug_npy_prefix}_{cam}"
            for k, v in features.items():
                np.save(f"{prefix}_{k}.npy", v)


def cmd_layout(args):
    make_vertical_test_output(
        base_dir=Path(args.base_dir),
        out_path=Path(args.out),
        pair_index=args.index,
        target_width=args.target_width,
        target_height=args.target_height,
        max_duration=args.seconds,
    )


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Unified dashcam pipeline runner",
    )
    p.add_argument("--base-dir", required=True)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--road-dir", default="DCIMA")
    s.add_argument("--cabin-dir", default="DCIMB")
    s.add_argument("--show", type=int, default=10)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("motion")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--camera", choices=["road", "cabin", "both"], default="both")
    s.add_argument("--downscale-width", type=int, default=320)
    s.add_argument("--frame-step", type=int, default=2)
    s.add_argument("--pre-event-sec", type=float, default=3.0)
    s.add_argument("--post-event-sec", type=float, default=5.0)
    s.add_argument("--min-event-gap-sec", type=float, default=5.0)
    s.add_argument("--threshold-std", type=float, default=2.0)
    s.add_argument("--max-events", type=int, default=20)
    s.set_defaults(func=cmd_motion)

    s = sub.add_parser("audio")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--camera", choices=["road", "cabin", "both"], default="road")
    s.add_argument("--sample-rate", type=int, default=16000)
    s.add_argument("--k", type=float, default=1.8)
    s.add_argument("--pre-event-sec", type=float, default=1.5)
    s.add_argument("--post-event-sec", type=float, default=2.5)
    s.add_argument("--min-event-gap-sec", type=float, default=1.0)
    s.add_argument("--max-events", type=int, default=10)
    s.add_argument("--debug-npy-prefix")
    s.set_defaults(func=cmd_audio)

    s = sub.add_parser("layout")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--out", default="output/test_vertical.mp4")
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--target-width", type=int, default=1080)
    s.add_argument("--target-height", type=int, default=1920)
    s.set_defaults(func=cmd_layout)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
    