from __future__ import annotations

from pathlib import Path
from typing import Optional

from moviepy import VideoFileClip, clips_array

from .ingest import DashcamConfig, discover_pairs


def make_vertical_test_output(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    target_width: int = 1080,
    target_height: int = 1920,
    max_duration: Optional[float] = None,
):
    """
    Load one road+cabin pair and create a vertical split screen test video.

    - base_dir: path to DCIM (folder that contains DCIMA and DIMCB).
    - out_path: where to write the MP4.
    - pair_index: which pair to use (0 = first).
    - max_duration: if set, trims the output to this many seconds.
    """
    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)

    if not pairs:
        raise RuntimeError("No clip pairs found — check ingest.")
    if pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    print(f"Using pair #{pair_index}:")
    print("  road :", pair.road)
    print("  cabin:", pair.cabin)

    road = VideoFileClip(str(pair.road))
    cabin = VideoFileClip(str(pair.cabin))

    if max_duration is not None:
        road = road.subclipped(0, max_duration)
        cabin = cabin.subclipped(0, max_duration)

    # Resize both to same width
    road_resized = road.resized(width=target_width)
    cabin_resized = cabin.resized(width=target_width)

    # Stack vertically (road on top, cabin on bottom)
    stacked = clips_array([
        [road_resized],
        [cabin_resized],
    ])

    # Force final size to 1080x1920
    stacked = stacked.resized((target_width, target_height))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering test composite to: {out_path}")
    stacked.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        threads=4,
        preset="medium",
    )

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vertical split-screen test renderer")
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to DCIM folder containing DCIMA and DIMCB",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/test/test_vertical.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which clip pair index to use (0-based)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Optional: trim to this many seconds for faster tests (default 10)",
    )
    args = parser.parse_args()

    make_vertical_test_output(
        base_dir=Path(args.base_dir),
        out_path=Path(args.out),
        pair_index=args.index,
        max_duration=args.seconds,
    )
