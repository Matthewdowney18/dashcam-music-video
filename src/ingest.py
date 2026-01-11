from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


# 👇 Edit this to wherever you copy/mount the SD card contents
# Example: Path("/media/youruser/63 GB Volume/DCIM")
DEFAULT_BASE_DIR = Path("/media/matt/2004-1014/DCIM")

DEFAULT_ROAD_DIR_NAME = "DCIMA"   # exterior folder
DEFAULT_CABIN_DIR_NAME = "DCIMB"  # interior folder (change if needed)
DEFAULT_EXTENSIONS = (".avi", ".mp4", ".mov")


@dataclass(frozen=True)
class DashcamConfig:
    base_dir: Path
    road_dir_name: str = DEFAULT_ROAD_DIR_NAME
    cabin_dir_name: str = DEFAULT_CABIN_DIR_NAME
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS

    @property
    def road_dir(self) -> Path:
        return self.base_dir / self.road_dir_name

    @property
    def cabin_dir(self) -> Path:
        return self.base_dir / self.cabin_dir_name


@dataclass(frozen=True)
class ClipPair:
    """Represents one paired minute: road + cabin clip."""
    road: Path
    cabin: Path

    def __str__(self) -> str:
        return f"{self.road.name}  |  {self.cabin.name}"


def _iter_video_files(directory: Path, extensions: Iterable[str]) -> List[Path]:
    """Return all video files in a directory with given extensions, sorted by name."""
    files: List[Path] = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def _clip_key(path: Path) -> str:
    """
    Extract a matching key from a filename, e.g.:

    MOVA0624.avi -> "0624"
    MOVB0624.avi -> "0624"
    ABCD-00123.avi -> "00123"

    If no trailing digits are found, fall back to the stem.
    """
    stem = path.stem  # e.g. "MOVA0624"
    m = re.search(r"(\d+)$", stem)
    if m:
        return m.group(1)
    return stem


def discover_pairs(config: DashcamConfig) -> List[ClipPair]:
    """
    Scan road + cabin folders and return a list of ClipPair objects,
    matching files by numeric ID (e.g. MOVA0624 <-> MOVB0624).
    """
    if not config.road_dir.is_dir():
        raise FileNotFoundError(f"Road dir not found: {config.road_dir}")

    if not config.cabin_dir.is_dir():
        raise FileNotFoundError(f"Cabin dir not found: {config.cabin_dir}")

    road_files = _iter_video_files(config.road_dir, config.extensions)
    cabin_files = _iter_video_files(config.cabin_dir, config.extensions)

    road_index = {}
    for p in road_files:
        key = _clip_key(p)
        road_index[key] = p

    cabin_index = {}
    for p in cabin_files:
        key = _clip_key(p)
        cabin_index[key] = p

    # Intersection of keys gives us the pairs
    common_keys = sorted(set(road_index.keys()) & set(cabin_index.keys()))

    pairs: List[ClipPair] = []
    for key in common_keys:
        pairs.append(ClipPair(road=road_index[key], cabin=cabin_index[key]))

    # (Optional) you could log keys that are only road or only cabin if you care.

    return pairs



# ---- CLI / debug entrypoint -------------------------------------------------


def _print_summary(pairs: List[ClipPair], config: DashcamConfig) -> None:
    print("Dashcam ingest summary")
    print("======================")
    print(f"Base dir : {config.base_dir}")
    print(f"Road dir : {config.road_dir}")
    print(f"Cabin dir: {config.cabin_dir}")
    print(f"Total paired clips: {len(pairs)}\n")

    show_n = min(10, len(pairs))
    if show_n:
        print(f"First {show_n} pairs:")
        for p in pairs[:show_n]:
            print("  ", p)
    else:
        print("No pairs found. Check folder names / extensions.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover paired dashcam clips (road + cabin)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="Path to DCIM folder containing DCIMA/DIMCB (default: %(default)s)",
    )
    parser.add_argument(
        "--road-dir",
        type=str,
        default=DEFAULT_ROAD_DIR_NAME,
        help="Subdirectory name for road/exterior camera.",
    )
    parser.add_argument(
        "--cabin-dir",
        type=str,
        default=DEFAULT_CABIN_DIR_NAME,
        help="Subdirectory name for cabin/interior camera.",
    )
    args = parser.parse_args()

    cfg = DashcamConfig(
        base_dir=Path(args.base_dir),
        road_dir_name=args.road_dir,
        cabin_dir_name=args.cabin_dir,
    )

    pairs = discover_pairs(cfg)
    _print_summary(pairs, cfg)
