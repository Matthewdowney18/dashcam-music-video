"""
layout.py

Video composition and layout utilities for producing debug and presentation
renders from dashcam clips.

This module defines how source videos are arranged into a single output frame
(e.g., inside+outside stacked, optional caption panel, debug overlays). It is
primarily responsible for generating FFmpeg filter graphs / commands that
implement the layout efficiently.

Typical responsibilities:
- Defining layout presets (stacked, side-by-side, caption area, etc.)
- Building FFmpeg filter_complex strings for scaling, padding, stacking, and text
- Keeping layout logic separate from event detection/scoring

Design goals:
- Use FFmpeg for composition (avoid Python-frame rendering)
- Make layouts reproducible and easy to tweak via named presets
- Keep render configuration explicit (dimensions, fonts, margins, etc.)

This module should NOT:
- Detect events (motion/audio) or choose segments
- Implement progress parsing (handled by layout_progress)
"""


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



# src/layout.py (additions)



# src/layout.py (additions / new function)


from pathlib import Path
from typing import List, Optional, Dict

from .ingest import DashcamConfig, discover_pairs
from .conicals import OverlayEvent, Lane, build_conical_lanes
from .layout_progress import _ffprobe_duration_seconds, _run_ffmpeg_with_progress

# assumes PRESETS already exists in layout.py (debug720 etc.)

def _escape_drawtext(text: str) -> str:
    # Escape ffmpeg drawtext special chars: \ : ' % ,
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
    )


def _ffprobe_video_width(path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out) if out.isdigit() else None


def _ffprobe_video_height(path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=height",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out) if out.isdigit() else None


def _make_captioned_output_preset(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    preset: str = "debug720",
    max_duration: Optional[float] = 60.0,
    panel_width: int = 0,
    stack_width: Optional[int] = None,
    detect_camera: str = "both",
    quality: str = "final",
):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESETS)}")
    if detect_camera not in {"road", "cabin", "both"}:
        raise ValueError("detect_camera must be one of: road, cabin, both")
    if panel_width < 0:
        raise ValueError("panel_width must be >= 0")

    p = PRESETS[preset]
    if p.caption_height <= 0:
        raise ValueError("Preset must include a caption band (caption_height > 0).")

    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)
    if not pairs:
        raise RuntimeError("No clip pairs found — check ingest.")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    road = pair.road
    cabin = pair.cabin

    # --- detect events ---
    lanes = build_conical_lanes(
        road=road,
        cabin=cabin,
        detect_camera=detect_camera,
        max_duration=max_duration,
    )

    # --- build filter graph ---
    target_width = p.width
    target_height = p.height
    caption_height = p.caption_height
    fps = p.fps

    out_path.parent.mkdir(parents=True, exist_ok=True)

    filters: List[str] = []
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    if quality not in {"debug", "final"}:
        raise ValueError("quality must be one of: debug, final")

    if panel_width > 0:
        road_w = _ffprobe_video_width(road)
        road_h = _ffprobe_video_height(road)
        cabin_w = _ffprobe_video_width(cabin)
        cabin_h = _ffprobe_video_height(cabin)
        if quality == "debug":
            panel_width = 420 if panel_width == 0 else panel_width
            stack_w = stack_width or 960
        else:
            stack_w = stack_width or road_w or target_width
        stack_h = None
        if road_w and road_h and cabin_w and cabin_h:
            road_scaled_h = int(round(road_h * (stack_w / road_w)))
            cabin_scaled_h = int(round(cabin_h * (stack_w / cabin_w)))
            stack_h = max(2, road_scaled_h) + max(2, cabin_scaled_h)

        filters.append(
            f"[0:v]scale={stack_w}:-2:force_original_aspect_ratio=decrease,setsar=1"
            f",pad={stack_w}:ih:(ow-iw)/2:(oh-ih)/2:black[v0]"
        )
        filters.append(
            f"[1:v]scale={stack_w}:-2:force_original_aspect_ratio=decrease,setsar=1"
            f",pad={stack_w}:ih:(ow-iw)/2:(oh-ih)/2:black[v1]"
        )
        filters.append("[v0][v1]vstack=inputs=2[vstack]")
        filters.append(f"[vstack]pad={stack_w + panel_width}:ih:0:0:black[vpad]")
        panel_alpha = 1.0 if quality == "debug" else 0.35
        filters.append(
            f"[vpad]drawbox=x={stack_w}:y=0:w={panel_width}:h=ih:color=black@{panel_alpha}:t=fill[vbase]"
        )

        panel_x0 = stack_w
        panel_y0 = 0
        panel_h = stack_h
        panel_w = panel_width
        panel_font_size = 18
        row_h = max(34, int(panel_font_size * 2.0))
        y_pad = max(2, int(panel_font_size * 0.2))
        margin_x = 18
        duration_s = None
        if max_duration is not None:
            duration_s = float(max_duration)
        else:
            d0 = _ffprobe_duration_seconds(road)
            d1 = _ffprobe_duration_seconds(cabin)
            duration_s = min(d0, d1) if (d0 and d1) else (d0 or d1)
        duration_s = duration_s if (duration_s and duration_s > 0) else 1.0
        timeline_w = max(1.0, float(panel_w - (margin_x * 2)))

        def time_to_x(t: float) -> float:
            t = max(0.0, min(float(t), duration_s))
            return panel_x0 + margin_x + (t / duration_s) * timeline_w

        def add_lane(lane: List[OverlayEvent], row_index: int, tag_in: str, header: str, text_events: bool) -> str:
            header_y = panel_y0 + row_index * row_h + y_pad
            timeline_y = panel_y0 + row_index * row_h + y_pad + int(row_h * 0.55)
            cur = tag_in
            if header:
                out_header = f"v_lane_{row_index}_hdr"
                safe_header = _escape_drawtext(header)
                filters.append(
                    f"{cur}drawtext=fontfile={fontfile}:"
                    f"text='{safe_header}':x={panel_x0 + margin_x}:y={header_y}:"
                    f"fontsize={panel_font_size}:fontcolor=white:"
                    f"enable='1'"
                    f"[{out_header}]"
                )
                cur = f"[{out_header}]"
            for k, e in enumerate(lane):
                x_start = time_to_x(e.start)
                x_end = time_to_x(e.end)
                x_peak = time_to_x(e.peak)
                w_box = max(2.0, x_end - x_start)
                h_box = max(6, int(row_h * 0.3))
                out_box = f"v_lane_{row_index}_{k}_box"
                filters.append(
                    f"{cur}drawbox=x={x_start:.2f}:y={timeline_y}:w={w_box:.2f}:h={h_box}:"
                    f"color=white@0.7:t=fill:"
                    f"enable='between(t,{e.start:.3f},{e.end:.3f})'"
                    f"[{out_box}]"
                )
                cur = f"[{out_box}]"
                if text_events:
                    out_tag = f"v_lane_{row_index}_{k}"
                    safe_label = _escape_drawtext(e.label)
                    filters.append(
                        f"{cur}drawtext=fontfile={fontfile}:"
                        f"text='{safe_label}':x={panel_x0 + margin_x}:y={header_y}:"
                        f"fontsize={panel_font_size}:fontcolor=white:"
                        f"enable='between(t,{e.start:.3f},{e.end:.3f})'"
                        f"[{out_tag}]"
                    )
                    cur = f"[{out_tag}]"

                out_tag2 = f"v_lane_{row_index}_{k}_dot"
                dot_start = max(0.0, e.peak - 0.15)
                dot_end = e.peak + 0.15
                filters.append(
                    f"{cur}drawbox=x={x_peak - 9:.2f}:y={timeline_y}:w=18:h=18:"
                    f"color=red@1.0:t=fill:"
                    f"enable='between(t,{dot_start:.3f},{dot_end:.3f})'"
                    f"[{out_tag2}]"
                )
                cur = f"[{out_tag2}]"
            return cur

        vtag = "[vbase]"
        debug_text_events = (quality == "final")
        for idx, lane in enumerate(lanes):
            allow_event_text = debug_text_events and lane.key != "score"
            vtag = add_lane(lane.events, idx, vtag, lane.title, allow_event_text)
        filters.append(f"{vtag}fps={fps}[vfinal]")
    else:
        video_height = target_height - caption_height
        half_h = video_height // 2
        video_height_exact = half_h * 2

        filters.append(f"[0:v]scale={target_width}:{half_h}:flags=lanczos,setsar=1[v0]")
        filters.append(f"[1:v]scale={target_width}:{half_h}:flags=lanczos,setsar=1[v1]")
        filters.append("[v0][v1]vstack=inputs=2[vstack]")

        if video_height_exact != video_height:
            filters.append(f"[vstack]pad={target_width}:{video_height}:0:0:black[vpad]")
            v_main = "[vpad]"
        else:
            v_main = "[vstack]"

        filters.append(f"{v_main}pad={target_width}:{target_height}:0:0:black[vbase]")

        band_y = target_height - caption_height
        row_h = max(1, caption_height // 4)  # 4 rows now

        def add_lane(lane: List[OverlayEvent], row_index: int, tag_in: str) -> str:
            y_text = band_y + row_index * row_h + int(row_h * 0.18)
            y_dot = band_y + row_index * row_h + int(row_h * 0.62)
            cur = tag_in
            for k, e in enumerate(lane):
                out_tag = f"v_lane_{row_index}_{k}"
                safe_label = _escape_drawtext(e.label)
                filters.append(
                    f"{cur}drawtext=fontfile={fontfile}:"
                    f"text='{safe_label}':x=24:y={y_text}:"
                    f"fontsize={int(row_h*0.55)}:fontcolor=white:"
                    f"enable='between(t,{e.start:.3f},{e.end:.3f})'"
                    f"[{out_tag}]"
                )
                cur = f"[{out_tag}]"

                out_tag2 = f"v_lane_{row_index}_{k}_dot"
                dot_start = max(0.0, e.peak - 0.15)
                dot_end = e.peak + 0.15
                filters.append(
                    f"{cur}drawbox=x={target_width - 60}:y={y_dot}:w=18:h=18:"
                    f"color=red@1.0:t=fill:"
                    f"enable='between(t,{dot_start:.3f},{dot_end:.3f})'"
                    f"[{out_tag2}]"
                )
                cur = f"[{out_tag2}]"
            return cur

        vtag = "[vbase]"
        for idx, lane in enumerate(lanes):
            vtag = add_lane(lane.events, idx, vtag)
        filters.append(f"{vtag}fps={fps}[vfinal]")

    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error", "-nostats",
        "-progress", "pipe:1",
        "-i", str(road),
        "-i", str(cabin),
        "-filter_complex", filter_complex,
        "-map", "[vfinal]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", ("ultrafast" if quality == "debug" else p.x264_preset),
        "-crf", str(23 if quality == "debug" else p.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
    ]
    if max_duration is not None:
        cmd += ["-t", str(max_duration)]
    cmd += [str(out_path)]

    expected_duration_s: Optional[float] = float(max_duration) if max_duration is not None else None
    if expected_duration_s is None:
        d0 = _ffprobe_duration_seconds(road)
        d1 = _ffprobe_duration_seconds(cabin)
        expected_duration_s = min(d0, d1) if (d0 and d1) else (d0 or d1)

    _run_ffmpeg_with_progress(cmd, expected_duration_s)


def make_vertical_captioned_output_preset(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    preset: str = "debug720",
    max_duration: Optional[float] = 60.0,
    detect_camera: str = "both",
):
    return _make_captioned_output_preset(
        base_dir=base_dir,
        out_path=out_path,
        pair_index=pair_index,
        preset=preset,
        max_duration=max_duration,
        panel_width=0,
        detect_camera=detect_camera,
    )


def make_captioned_output_preset(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    preset: str = "debug720",
    max_duration: Optional[float] = 60.0,
    panel_width: Optional[int] = None,
    detect_camera: str = "both",
    stack_width: Optional[int] = None,
    quality: str = "debug",
):
    if panel_width is None:
        panel_width = 420 if quality == "debug" else 480
    return _make_captioned_output_preset(
        base_dir=base_dir,
        out_path=out_path,
        pair_index=pair_index,
        preset=preset,
        max_duration=max_duration,
        panel_width=panel_width,
        detect_camera=detect_camera,
        stack_width=stack_width,
        quality=quality,
    )
