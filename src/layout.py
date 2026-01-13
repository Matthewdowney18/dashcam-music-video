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


from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import math

from .ingest import DashcamConfig, discover_pairs
from .motion_events import detect_motion_events
from .audio_events import detect_audio_events_for_clip
from .layout_progress import _ffprobe_duration_seconds, _run_ffmpeg_with_progress

# assumes PRESETS already exists in layout.py (debug720 etc.)

@dataclass(frozen=True)
class OverlayEvent:
    start: float
    end: float
    peak: float
    label: str
    strength01: float = 1.0  # normalized strength (0..1)


def _merge_overlapping(events: List[OverlayEvent]) -> List[OverlayEvent]:
    if not events:
        return []
    evs = sorted(events, key=lambda e: (e.start, e.end))
    merged: List[OverlayEvent] = []
    cur = evs[0]
    for e in evs[1:]:
        if e.start <= cur.end:
            new_start = min(cur.start, e.start)
            new_end = max(cur.end, e.end)
            mid = (new_start + new_end) / 2.0
            new_peak = min([cur.peak, e.peak], key=lambda p: abs(p - mid))
            # keep the stronger one
            new_strength = max(cur.strength01, e.strength01)
            cur = OverlayEvent(new_start, new_end, new_peak, cur.label, new_strength)
        else:
            merged.append(cur)
            cur = e
    merged.append(cur)
    return merged


def _normalize_strength(values: List[float]) -> List[float]:
    # normalize to 0..1 per-clip, stable for small lists
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _build_union_windows(*lanes: List[OverlayEvent]) -> List[Tuple[float, float]]:
    # union of all intervals from lanes
    intervals: List[Tuple[float, float]] = []
    for lane in lanes:
        for e in lane:
            intervals.append((e.start, e.end))
    if not intervals:
        return []
    intervals.sort()
    out: List[Tuple[float, float]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def _lane_strength_in_window(lane: List[OverlayEvent], start: float, end: float) -> float:
    # max strength among events that overlap window
    best = 0.0
    for e in lane:
        if e.end >= start and e.start <= end:
            best = max(best, e.strength01)
    return best


def _best_peak_time_in_window(lane: List[OverlayEvent], start: float, end: float) -> Optional[float]:
    # pick peak of strongest overlapping event in lane
    best_strength = -1.0
    best_peak = None
    for e in lane:
        if e.end >= start and e.start <= end:
            if e.strength01 > best_strength:
                best_strength = e.strength01
                best_peak = e.peak
    return best_peak


def _compute_score_events(
    road_lane: List[OverlayEvent],
    cabin_lane: List[OverlayEvent],
    audio_lane: List[OverlayEvent],
) -> List[OverlayEvent]:
    windows = _build_union_windows(road_lane, cabin_lane, audio_lane)
    score_events: List[OverlayEvent] = []

    for (s, e) in windows:
        mr = _lane_strength_in_window(road_lane, s, e)
        mc = _lane_strength_in_window(cabin_lane, s, e)
        au = _lane_strength_in_window(audio_lane, s, e)

        score = (
            0.40 * mr
            + 0.30 * mc
            + 0.30 * au
            + 0.20 * math.sqrt(mr * au + 1e-9)
            + 0.10 * math.sqrt(mc * au + 1e-9)
        )
        score = max(0.0, min(1.0, score))

        # peak time: prefer strongest lane’s peak within window; fall back to middle
        peak_candidates = [
            _best_peak_time_in_window(road_lane, s, e),
            _best_peak_time_in_window(cabin_lane, s, e),
            _best_peak_time_in_window(audio_lane, s, e),
        ]
        peak_candidates = [p for p in peak_candidates if p is not None]
        peak = peak_candidates[0] if peak_candidates else (s + e) / 2.0

        score_events.append(
            OverlayEvent(
                start=s,
                end=e,
                peak=peak,
                label=f"SCORE {score:.2f}",
                strength01=score,
            )
        )

    # optional: merge to avoid label spam
    return _merge_overlapping(score_events)


def make_vertical_captioned_output_preset(
    base_dir: Path,
    out_path: Path,
    pair_index: int = 0,
    preset: str = "debug720",
    max_duration: Optional[float] = 60.0,
):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESETS)}")
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
    road_motion = detect_motion_events(video_path=road, downscale_width=320, frame_step=2,
                                      pre_event_sec=3.0, post_event_sec=5.0, min_event_gap_sec=5.0,
                                      threshold_std=2.0, max_events=20)
    cabin_motion = detect_motion_events(video_path=cabin, downscale_width=320, frame_step=2,
                                       pre_event_sec=3.0, post_event_sec=5.0, min_event_gap_sec=5.0,
                                       threshold_std=2.0, max_events=20)

    audio_events, _features = detect_audio_events_for_clip(
        video_path=road,  # audio is same for both in your setup; choose road
        sample_rate=16000,
        k=1.8,
        pre_event_sec=1.5,
        post_event_sec=2.5,
        min_event_gap_sec=1.0,
        max_events=10,
    )

    # normalize strengths per lane
    road_strengths = _normalize_strength([float(e.score) for e in road_motion])
    cabin_strengths = _normalize_strength([float(e.score) for e in cabin_motion])
    audio_strengths = _normalize_strength([float(e.peak_score) for e in audio_events])

    road_lane = _merge_overlapping([
        OverlayEvent(float(e.start_time), float(e.end_time), float(e.peak_time), "MOTION ROAD", road_strengths[i])
        for i, e in enumerate(road_motion)
    ])
    cabin_lane = _merge_overlapping([
        OverlayEvent(float(e.start_time), float(e.end_time), float(e.peak_time), "MOTION CABIN", cabin_strengths[i])
        for i, e in enumerate(cabin_motion)
    ])
    audio_lane = _merge_overlapping([
        OverlayEvent(float(e.start), float(e.end), float(e.peak_time), "AUDIO", audio_strengths[i])
        for i, e in enumerate(audio_events)
    ])

    score_lane = _compute_score_events(road_lane, cabin_lane, audio_lane)

    # trim to max_duration
    if max_duration is not None:
        def clip_lane(lane: List[OverlayEvent]) -> List[OverlayEvent]:
            out = []
            for e in lane:
                if e.start >= max_duration:
                    continue
                out.append(
                    OverlayEvent(
                        start=max(0.0, e.start),
                        end=min(float(max_duration), e.end),
                        peak=min(float(max_duration), max(0.0, e.peak)),
                        label=e.label,
                        strength01=e.strength01,
                    )
                )
            return out
        road_lane = clip_lane(road_lane)
        cabin_lane = clip_lane(cabin_lane)
        audio_lane = clip_lane(audio_lane)
        score_lane = clip_lane(score_lane)

    # --- build filter graph ---
    target_width = p.width
    target_height = p.height
    caption_height = p.caption_height
    fps = p.fps

    video_height = target_height - caption_height
    half_h = video_height // 2
    video_height_exact = half_h * 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    filters: List[str] = []
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
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def add_lane(lane: List[OverlayEvent], row_index: int, tag_in: str) -> str:
        y_text = band_y + row_index * row_h + int(row_h * 0.18)
        y_dot = band_y + row_index * row_h + int(row_h * 0.62)
        cur = tag_in
        for k, e in enumerate(lane):
            out_tag = f"v_lane_{row_index}_{k}"
            safe_label = e.label.replace(":", "\\:")
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
    vtag = add_lane(road_lane, 0, vtag)
    vtag = add_lane(cabin_lane, 1, vtag)
    vtag = add_lane(audio_lane, 2, vtag)
    vtag = add_lane(score_lane, 3, vtag)

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
        "-preset", p.x264_preset,
        "-crf", str(p.crf),
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
