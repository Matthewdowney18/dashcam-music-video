from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .audio_events import detect_audio_events_for_clip
from .motion_events import detect_motion_events


@dataclass(frozen=True)
class OverlayEvent:
    start: float
    end: float
    peak: float
    label: str
    strength01: float = 1.0  # normalized strength (0..1)


@dataclass(frozen=True)
class Lane:
    key: str
    title: str
    events: List[OverlayEvent]


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


def build_conical_lanes(
    road: Path,
    cabin: Path,
    detect_camera: str = "both",
    max_duration: Optional[float] = None,
) -> List[Lane]:
    road_motion = []
    cabin_motion = []
    if detect_camera in {"road", "both"}:
        road_motion = detect_motion_events(
            video_path=road,
            downscale_width=320,
            frame_step=2,
            pre_event_sec=3.0,
            post_event_sec=5.0,
            min_event_gap_sec=5.0,
            threshold_std=2.0,
            max_events=20,
        )
    if detect_camera in {"cabin", "both"}:
        cabin_motion = detect_motion_events(
            video_path=cabin,
            downscale_width=320,
            frame_step=2,
            pre_event_sec=3.0,
            post_event_sec=5.0,
            min_event_gap_sec=5.0,
            threshold_std=2.0,
            max_events=20,
        )

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

    return [
        Lane(key="motion_road", title="MOTION ROAD", events=road_lane),
        Lane(key="motion_cabin", title="MOTION CABIN", events=cabin_lane),
        Lane(key="audio", title="AUDIO", events=audio_lane),
        Lane(key="score", title="SCORE", events=score_lane),
    ]
