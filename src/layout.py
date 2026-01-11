from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from .ingest import DashcamConfig, discover_pairs
from .layout_progress import _ffprobe_duration_seconds, _run_ffmpeg_with_progress

@dataclass(frozen=True)
class LayoutPreset:
    name: str
    width: int
    height: int
    caption_height: int  # 0 = no caption band
    fps: int
    x264_preset: str
    crf: int  # quality (lower = better, bigger file). 18-28 typical.


PRESETS: Dict[str, LayoutPreset] = {
    # Matches your original "force to 1080x1920" behavior (no reserved caption band).
    # Still uses exact sizing for each half to avoid final squish step.
    "legacy1080": LayoutPreset(
        name="legacy1080",
        width=1080,
        height=1920,
        caption_height=0,
        fps=30,
        x264_preset="veryfast",
        crf=20,
    ),
    # 1080x1920 with a reserved caption band at the bottom
    "caption1080": LayoutPreset(
        name="caption1080",
        width=1080,
        height=1920,
        caption_height=160,
        fps=30,
        x264_preset="veryfast",
        crf=20,
    ),
    # Faster debug presets
    "debug720": LayoutPreset(
        name="debug720",
        width=720,
        height=1280,
        caption_height=120,
        fps=24,
        x264_preset="veryfast",
        crf=22,
    ),
    "debug540": LayoutPreset(
        name="debug540",
        width=540,
        height=960,
        caption_height=96,
        fps=24,
        x264_preset="veryfast",
        crf=24,
    ),
}




def make_vertical_test_output(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    target_width: int = 1080,
    target_height: int = 1920,
    caption_height: int = 0,
    max_duration: Optional[float] = None,
    fps: int = 30,
    x264_preset: str = "veryfast",
    crf: int = 20,
):
    """
    FFmpeg-based compositor: road (top) + cabin (bottom) + optional caption band.

    - caption_height: reserved blank band at bottom (px). 0 disables it.
    - max_duration: trims output to N seconds if provided.
    - fps/x264_preset/crf: encoding knobs.
    """
    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)

    if not pairs:
        raise RuntimeError("No clip pairs found — check ingest.")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    print(f"Using pair #{pair_index}:")
    print("  road :", pair.road)
    print("  cabin:", pair.cabin)

    if caption_height < 0:
        raise ValueError("caption_height must be >= 0")

    video_height = target_height - caption_height
    if video_height <= 0:
        raise ValueError(
            f"caption_height={caption_height} leaves no room for video "
            f"(target_height={target_height})"
        )

    half_h = video_height // 2
    # If video_height is odd, we’ll pad the last line in the stack automatically
    # by scaling both halves to half_h and then padding the vstack to video_height.
    # This keeps things stable.
    video_height_exact = half_h * 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter graph:
    # - scale each input to (W x half_h)
    # - vstack into W x (2*half_h)
    # - pad to W x video_height (handles odd heights)
    # - optionally extend canvas downward for caption band (black)
    # - fps filter
    #
    # Note: we use setsar=1 to avoid weird aspect ratio flags.
    filters = []

    filters.append(
        f"[0:v]scale={target_width}:{half_h}:flags=lanczos,setsar=1[v0]"
    )
    filters.append(
        f"[1:v]scale={target_width}:{half_h}:flags=lanczos,setsar=1[v1]"
    )
    filters.append("[v0][v1]vstack=inputs=2[vstack]")

    # Pad vstack to exact intended video area (in case of odd target_height-caption_height)
    if video_height_exact != video_height:
        filters.append(
            f"[vstack]pad={target_width}:{video_height}:0:0:black[vpad]"
        )
        v_main = "[vpad]"
    else:
        v_main = "[vstack]"

    if caption_height > 0:
        # Extend canvas downward by caption_height (adds black band)
        filters.append(
            f"{v_main}pad={target_width}:{target_height}:0:0:black[vout]"
        )
    else:
        # Ensure final is exactly target_height (legacy: no band, but guarantee size)
        filters.append(
            f"{v_main}pad={target_width}:{target_height}:0:0:black[vout]"
        )

    filters.append(f"[vout]fps={fps}[vfinal]")

    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-progress", "pipe:1",
        "-i",
        str(pair.road),
        "-i",
        str(pair.cabin),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vfinal]",
        # audio: take from road by default (can change later)
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        x264_preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
    ]

    if max_duration is not None:
        # Output trim (fast, simple). For perfect accuracy you can do input trims instead.
        cmd += ["-t", str(max_duration)]

    cmd += [str(out_path)]

    print(f"Rendering test composite to: {out_path}")

    # Estimate duration for progress bar
    expected_duration_s: Optional[float] = None
    if max_duration is not None:
        expected_duration_s = float(max_duration)
    else:
        d0 = _ffprobe_duration_seconds(pair.road)
        d1 = _ffprobe_duration_seconds(pair.cabin)
        if d0 and d1:
            expected_duration_s = min(d0, d1)
        else:
            expected_duration_s = d0 or d1

    try:
        _run_ffmpeg_with_progress(cmd, expected_duration_s)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.") from e

    print("Done.")



def make_vertical_test_output_preset(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    preset: str = "caption1080",
    max_duration: Optional[float] = None,
):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESETS)}")
    p = PRESETS[preset]
    return make_vertical_test_output(
        base_dir=base_dir,
        out_path=out_path,
        pair_index=pair_index,
        target_width=p.width,
        target_height=p.height,
        caption_height=p.caption_height,
        max_duration=max_duration,
        fps=p.fps,
        x264_preset=p.x264_preset,
        crf=p.crf,
    )



