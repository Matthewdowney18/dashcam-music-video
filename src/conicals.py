from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .audio_events import detect_audio_events_for_clip
from .motion_events import detect_motion_events

try:
    import tomllib
except Exception:  # pragma: no cover - fallback when tomllib unavailable
    tomllib = None


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


@dataclass(frozen=True)
class LaneSpec:
    key: str
    title: str
    enabled: bool
    detector: str
    camera: str
    params: Dict[str, object]


DEFAULT_LANE_SPECS: List[Dict[str, object]] = [
    {
        "key": "motion_road",
        "title": "MOTION ROAD",
        "enabled": True,
        "detector": "motion",
        "camera": "road",
        "params": {
            "downscale_width": 320,
            "frame_step": 2,
            "pre_event_sec": 3.0,
            "post_event_sec": 5.0,
            "min_event_gap_sec": 5.0,
            "threshold_std": 2.0,
            "max_events": 20,
        },
    },
    {
        "key": "motion_cabin",
        "title": "MOTION CABIN",
        "enabled": True,
        "detector": "motion",
        "camera": "cabin",
        "params": {
            "downscale_width": 320,
            "frame_step": 2,
            "pre_event_sec": 3.0,
            "post_event_sec": 5.0,
            "min_event_gap_sec": 5.0,
            "threshold_std": 2.0,
            "max_events": 20,
        },
    },
    {
        "key": "audio",
        "title": "AUDIO",
        "enabled": True,
        "detector": "audio",
        "camera": "road",
        "params": {
            "sample_rate": 16000,
            "k": 1.8,
            "pre_event_sec": 1.5,
            "post_event_sec": 2.5,
            "min_event_gap_sec": 1.0,
            "max_events": 10,
        },
    },
    {
        "key": "score",
        "title": "SCORE",
        "enabled": True,
        "detector": "score",
        "camera": "both",
        "params": {
            "sources": ["motion_road", "motion_cabin", "audio"],
            "weights": {"motion_road": 0.4, "motion_cabin": 0.3, "audio": 0.3},
        },
    },
]


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


def _load_lane_specs(config_path: Optional[Path]) -> List[LaneSpec]:
    base_dir = Path(__file__).resolve().parents[1]
    candidates = []
    if config_path is not None:
        candidates.append(Path(config_path))
    else:
        candidates.append(base_dir / "config" / "conicals.toml")
        candidates.append(base_dir / "conicals.toml")
        candidates.append(base_dir / "config" / "conicals.json")
        candidates.append(base_dir / "conicals.json")

    data = None
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".toml" and tomllib is not None:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            break
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            break

    if data is None:
        raw_specs = DEFAULT_LANE_SPECS
    else:
        raw_specs = data.get("lane") or data.get("lanes") or []

    specs: List[LaneSpec] = []
    for raw in raw_specs:
        key = str(raw.get("key", "")).strip()
        title = str(raw.get("title", "")).strip()
        detector = str(raw.get("detector", "")).strip()
        enabled = bool(raw.get("enabled", True))
        camera = str(raw.get("camera", "both")).strip()
        params = dict(raw.get("params", {}) or {})
        if not key or not title or not detector:
            continue
        if camera not in {"road", "cabin", "both"}:
            camera = "both"
        specs.append(
            LaneSpec(
                key=key,
                title=title,
                enabled=enabled,
                detector=detector,
                camera=camera,
                params=params,
            )
        )
    return specs


def _clip_lane(lane: List[OverlayEvent], max_duration: Optional[float]) -> List[OverlayEvent]:
    if max_duration is None:
        return lane
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


def _camera_allowed(spec_camera: str, detect_camera: str) -> bool:
    if detect_camera == "both":
        return True
    return spec_camera in {detect_camera, "both"}


def _build_motion_lane(spec: LaneSpec, road: Path, cabin: Path, detect_camera: str) -> List[OverlayEvent]:
    if not _camera_allowed(spec.camera, detect_camera):
        return []
    params = spec.params
    if spec.camera == "cabin":
        path = cabin
    elif spec.camera == "road":
        path = road
    else:
        path = road
    events = detect_motion_events(
        video_path=path,
        downscale_width=int(params.get("downscale_width", 320)),
        frame_step=int(params.get("frame_step", 2)),
        pre_event_sec=float(params.get("pre_event_sec", 3.0)),
        post_event_sec=float(params.get("post_event_sec", 5.0)),
        min_event_gap_sec=float(params.get("min_event_gap_sec", 5.0)),
        threshold_std=float(params.get("threshold_std", 2.0)),
        max_events=int(params.get("max_events", 20)),
    )
    strengths = _normalize_strength([float(e.score) for e in events])
    lane_events = [
        OverlayEvent(float(e.start_time), float(e.end_time), float(e.peak_time), spec.title, strengths[i])
        for i, e in enumerate(events)
    ]
    return _merge_overlapping(lane_events)


def _build_audio_lane(spec: LaneSpec, road: Path, cabin: Path, detect_camera: str) -> List[OverlayEvent]:
    if not _camera_allowed(spec.camera, detect_camera):
        return []
    params = spec.params
    path = cabin if spec.camera == "cabin" else road
    events, _features = detect_audio_events_for_clip(
        video_path=path,
        sample_rate=int(params.get("sample_rate", 16000)),
        k=float(params.get("k", 1.8)),
        pre_event_sec=float(params.get("pre_event_sec", 1.5)),
        post_event_sec=float(params.get("post_event_sec", 2.5)),
        min_event_gap_sec=float(params.get("min_event_gap_sec", 1.0)),
        max_events=int(params.get("max_events", 10)),
    )
    strengths = _normalize_strength([float(e.peak_score) for e in events])
    lane_events = [
        OverlayEvent(float(e.start), float(e.end), float(e.peak_time), spec.title, strengths[i])
        for i, e in enumerate(events)
    ]
    return _merge_overlapping(lane_events)


def _build_score_lane(spec: LaneSpec, sources: Dict[str, List[OverlayEvent]]) -> List[OverlayEvent]:
    src_keys = list(spec.params.get("sources", []))
    if not src_keys:
        return []
    weights_raw = dict(spec.params.get("weights", {}) or {})
    total_w = 0.0
    for k in src_keys:
        total_w += float(weights_raw.get(k, 1.0))
    total_w = total_w if total_w > 0 else 1.0

    lanes = [sources.get(k, []) for k in src_keys]
    windows = _build_union_windows(*lanes)
    score_events: List[OverlayEvent] = []
    for (s, e) in windows:
        best_peak = None
        best_strength = -1.0
        score = 0.0
        for k in src_keys:
            lane = sources.get(k, [])
            w = float(weights_raw.get(k, 1.0))
            strength = _lane_strength_in_window(lane, s, e)
            score += w * strength
            if strength * w > best_strength:
                best_strength = strength * w
                best_peak = _best_peak_time_in_window(lane, s, e)
        score = max(0.0, min(1.0, score / total_w))
        peak = best_peak if best_peak is not None else (s + e) / 2.0
        score_events.append(
            OverlayEvent(
                start=s,
                end=e,
                peak=peak,
                label=f"{spec.title} {score:.2f}",
                strength01=score,
            )
        )
    return _merge_overlapping(score_events)


def build_conical_lanes(
    road: Path,
    cabin: Path,
    detect_camera: str = "both",
    max_duration: Optional[float] = None,
    config_path: Optional[Path] = None,
) -> List[Lane]:
    specs = _load_lane_specs(config_path)
    registry: Dict[str, Callable[[LaneSpec], List[OverlayEvent]]] = {
        "motion": lambda spec: _build_motion_lane(spec, road, cabin, detect_camera),
        "audio": lambda spec: _build_audio_lane(spec, road, cabin, detect_camera),
    }

    built: Dict[str, List[OverlayEvent]] = {}
    for spec in specs:
        if not spec.enabled or spec.detector == "score":
            continue
        builder = registry.get(spec.detector)
        if builder is None:
            built[spec.key] = []
            continue
        built[spec.key] = _clip_lane(builder(spec), max_duration)

    for spec in specs:
        if not spec.enabled or spec.detector != "score":
            continue
        built[spec.key] = _clip_lane(_build_score_lane(spec, built), max_duration)

    lanes: List[Lane] = []
    for spec in specs:
        if not spec.enabled:
            continue
        lanes.append(Lane(key=spec.key, title=spec.title, events=built.get(spec.key, [])))
    return lanes
