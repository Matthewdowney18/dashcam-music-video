from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .audio_events import detect_audio_events_for_clip
from .motion_events import (
    detect_motion_events,
    detect_optical_flow,
    detect_vehicle_dynamics,
)
from .config_loader import ConfigLoader
from .detection_profiles import (
    MotionEventDetectConfig,
    OpticalFlowDetectConfig,
    VehicleDynamicsDetectConfig,
    get_flow_profile,
    get_motion_profile,
    get_vehicle_profile,
)


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
    profile: Optional[str]
    model: Optional[str]
    params: Dict[str, object]


DEFAULT_LANE_SPECS: List[Dict[str, object]] = [
    {
        "key": "motion_road",
        "title": "MOTION ROAD",
        "enabled": True,
        "detector": "motion",
        "camera": "road",
        "profile": "legacy_meanstd",
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
        "profile": "legacy_meanstd",
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
        "profile": "default",
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
        "key": "flow_road",
        "title": "FLOW ROAD",
        "enabled": True,
        "detector": "optical_flow",
        "camera": "road",
        "profile": "legacy_percentile",
        "params": {
            "downscale_width": 320,
            "frame_step": 2,
            "min_duration_sec": 2.0,
            "flow_percentile": 60.0,
        },
    },
    {
        "key": "flow_cabin",
        "title": "FLOW CABIN",
        "enabled": True,
        "detector": "optical_flow",
        "camera": "cabin",
        "profile": "legacy_percentile",
        "params": {
            "downscale_width": 320,
            "frame_step": 2,
            "min_duration_sec": 2.0,
            "flow_percentile": 60.0,
        },
    },
    {
        "key": "vehicle_moving_road",
        "title": "MOVING",
        "enabled": True,
        "detector": "vehicle_dynamics",
        "camera": "road",
        "profile": "default",
        "params": {"kind": "moving"},
    },
    {
        "key": "vehicle_stopped_road",
        "title": "STOPPED",
        "enabled": True,
        "detector": "vehicle_dynamics",
        "camera": "road",
        "profile": "default",
        "params": {"kind": "stopped"},
    },
    {
        "key": "vehicle_accel_road",
        "title": "ACCEL",
        "enabled": True,
        "detector": "vehicle_dynamics",
        "camera": "road",
        "profile": "default",
        "params": {"kind": "accel"},
    },
    {
        "key": "vehicle_decel_road",
        "title": "DECEL",
        "enabled": True,
        "detector": "vehicle_dynamics",
        "camera": "road",
        "profile": "default",
        "params": {"kind": "decel"},
    },
    {
        "key": "vehicle_turn_road",
        "title": "TURN",
        "enabled": True,
        "detector": "vehicle_dynamics",
        "camera": "road",
        "profile": "default",
        "params": {"kind": "turn"},
    },
    {
        "key": "score",
        "title": "SCORE",
        "enabled": True,
        "detector": "score",
        "camera": "both",
        "model": "weighted_union",
        "params": {},
    },
]

DEFAULT_SCORING_MODELS: Dict[str, Dict[str, object]] = {
    "weighted_union": {
        "sources": [
            "motion_road",
            "motion_cabin",
            "audio",
            "vehicle_accel_road",
            "vehicle_decel_road",
            "vehicle_turn_road",
        ],
        "weights": {
            "motion_road": 0.35,
            "motion_cabin": 0.25,
            "audio": 0.25,
            "vehicle_accel_road": 0.05,
            "vehicle_decel_road": 0.05,
            "vehicle_turn_road": 0.05,
        },
        "normalize": "sum",
        "peak_strategy": "best_weighted",
        "merge_overlaps": True,
    }
}


def _parse_scoring_models(data: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    models = data.get("model") if isinstance(data, dict) else None
    if not isinstance(models, dict) or not models:
        return DEFAULT_SCORING_MODELS
    return {str(k): dict(v) for k, v in models.items()}


def _parse_detection_overrides(
    data: Dict[str, object],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not isinstance(data, dict):
        return {"motion": {}, "flow": {}, "vehicle": {}}
    motion = data.get("motion") if isinstance(data.get("motion"), dict) else {}
    flow = data.get("flow") if isinstance(data.get("flow"), dict) else {}
    vehicle = data.get("vehicle") if isinstance(data.get("vehicle"), dict) else {}
    return {
        "motion": {str(k): dict(v) for k, v in motion.items()},
        "flow": {str(k): dict(v) for k, v in flow.items()},
        "vehicle": {str(k): dict(v) for k, v in vehicle.items()},
    }


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


def _validate_lane_keys(specs: List[LaneSpec]) -> None:
    seen: Dict[str, int] = {}
    for spec in specs:
        seen[spec.key] = seen.get(spec.key, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        dupes_str = ", ".join(sorted(dupes))
        raise ValueError(f"Lane keys must be unique. Duplicates: {dupes_str}")


def _validate_score_model(
    model_name: str,
    model_cfg: Dict[str, object],
    lane_keys: List[str],
) -> None:
    src_keys = [str(k) for k in model_cfg.get("sources", [])]
    if not src_keys:
        raise ValueError(f"Score model {model_name!r} must define sources.")
    missing = [k for k in src_keys if k not in lane_keys]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Score model {model_name!r} references unknown lanes: {missing_str}")
    weights_raw = dict(model_cfg.get("weights", {}) or {})
    extra_weights = [k for k in weights_raw.keys() if str(k) not in src_keys]
    if extra_weights:
        extra_str = ", ".join(sorted([str(k) for k in extra_weights]))
        raise ValueError(
            f"Score model {model_name!r} has weight keys not in sources: {extra_str}"
        )


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


def _load_lane_specs(
    lanes_data: Dict[str, object],
    legacy_config: bool,
) -> List[LaneSpec]:
    if not lanes_data:
        raw_specs = DEFAULT_LANE_SPECS
    else:
        raw_specs = lanes_data.get("lane") or lanes_data.get("lanes") or []

    specs: List[LaneSpec] = []
    for raw in raw_specs:
        key = str(raw.get("key", "")).strip()
        title = str(raw.get("title", "")).strip()
        detector = str(raw.get("detector", "")).strip()
        enabled = bool(raw.get("enabled", True))
        camera = str(raw.get("camera", "both")).strip()
        profile = raw.get("profile")
        model = raw.get("model")
        params = dict(raw.get("params", {}) or {})
        if not key or not title or not detector:
            continue
        if camera not in {"road", "cabin", "both"}:
            camera = "both"
        if profile is not None:
            profile = str(profile).strip() or None
        if model is not None:
            model = str(model).strip() or None
        if profile is None and legacy_config:
            if detector == "motion":
                profile = "legacy_meanstd"
            elif detector == "optical_flow":
                profile = "legacy_percentile"
            elif detector == "audio":
                profile = "default"
        specs.append(
            LaneSpec(
                key=key,
                title=title,
                enabled=enabled,
                detector=detector,
                camera=camera,
                profile=profile,
                model=model,
                params=params,
            )
        )
    _validate_lane_keys(specs)
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


def _resolve_motion_config(
    spec: LaneSpec,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
) -> MotionEventDetectConfig:
    profile_name = spec.profile or "default"
    profile_cfg = get_motion_profile(profile_name)
    profile_override = detection_overrides.get("motion", {}).get(profile_name, {})
    profile_cfg = MotionEventDetectConfig.from_dict(profile_override, base=profile_cfg)
    return MotionEventDetectConfig.from_dict(spec.params, base=profile_cfg)


def _resolve_flow_config(
    spec: LaneSpec,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
) -> OpticalFlowDetectConfig:
    profile_name = spec.profile or "default"
    profile_cfg = get_flow_profile(profile_name)
    profile_override = detection_overrides.get("flow", {}).get(profile_name, {})
    profile_cfg = OpticalFlowDetectConfig.from_dict(profile_override, base=profile_cfg)
    return OpticalFlowDetectConfig.from_dict(spec.params, base=profile_cfg)


def _resolve_vehicle_config(
    spec: LaneSpec,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
) -> VehicleDynamicsDetectConfig:
    profile_name = spec.profile or "default"
    profile_cfg = get_vehicle_profile(profile_name)
    profile_override = detection_overrides.get("vehicle", {}).get(profile_name, {})
    profile_cfg = VehicleDynamicsDetectConfig.from_dict(profile_override, base=profile_cfg)
    return VehicleDynamicsDetectConfig.from_dict(spec.params, base=profile_cfg)


def _build_motion_lane(
    spec: LaneSpec,
    road: Path,
    cabin: Path,
    detect_camera: str,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
) -> List[OverlayEvent]:
    if not _camera_allowed(spec.camera, detect_camera):
        return []
    cfg = _resolve_motion_config(spec, detection_overrides)
    if spec.camera == "cabin":
        path = cabin
    elif spec.camera == "road":
        path = road
    else:
        path = road
    events = detect_motion_events(video_path=path, config=cfg)
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


def _build_score_lane(
    spec: LaneSpec,
    sources: Dict[str, List[OverlayEvent]],
    model_cfg: Dict[str, object],
) -> List[OverlayEvent]:
    src_keys = [str(k) for k in model_cfg.get("sources", [])]
    if not src_keys:
        return []
    weights_raw = dict(model_cfg.get("weights", {}) or {})
    normalize = str(model_cfg.get("normalize", "sum")).strip().lower()
    peak_strategy = str(model_cfg.get("peak_strategy", "best_weighted")).strip().lower()
    merge_overlaps = bool(model_cfg.get("merge_overlaps", True))

    weights: Dict[str, float] = {}
    for k in src_keys:
        weights[k] = float(weights_raw.get(k, 1.0))

    lanes = [sources.get(k, []) for k in src_keys]
    windows = _build_union_windows(*lanes)
    score_events: List[OverlayEvent] = []
    for (s, e) in windows:
        best_peak = None
        best_strength = -1.0
        score = 0.0
        for k in src_keys:
            lane = sources.get(k, [])
            w = weights.get(k, 1.0)
            strength = _lane_strength_in_window(lane, s, e)
            score += w * strength
            weighted_strength = strength * w if peak_strategy == "best_weighted" else strength
            if weighted_strength > best_strength:
                best_strength = weighted_strength
                best_peak = _best_peak_time_in_window(lane, s, e)

        if normalize == "sum":
            total_w = sum(weights.values()) or 1.0
            score = score / total_w
        elif normalize == "max":
            max_w = max(weights.values()) if weights else 1.0
            score = score / (max_w or 1.0)
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode {normalize!r} for score model {spec.model!r}")

        score = max(0.0, min(1.0, score))
        if peak_strategy == "middle":
            peak = (s + e) / 2.0
        else:
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
    return _merge_overlapping(score_events) if merge_overlaps else score_events


def _build_optical_flow_lane(
    spec: LaneSpec,
    road: Path,
    cabin: Path,
    detect_camera: str,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
) -> List[OverlayEvent]:
    if not _camera_allowed(spec.camera, detect_camera):
        return []
    cfg = _resolve_flow_config(spec, detection_overrides)
    if spec.camera == "cabin":
        path = cabin
    elif spec.camera == "road":
        path = road
    else:
        path = road
    segments = detect_optical_flow(video_path=path, config=cfg)
    strengths = _normalize_strength([float(s.mean_flow) for s in segments])
    lane_events = [
        OverlayEvent(
            float(s.start_time),
            float(s.end_time),
            float((s.start_time + s.end_time) / 2.0),
            spec.title,
            strengths[i],
        )
        for i, s in enumerate(segments)
    ]
    return _merge_overlapping(lane_events)


def _build_vehicle_lane(
    spec: LaneSpec,
    road: Path,
    cabin: Path,
    detect_camera: str,
    detection_overrides: Dict[str, Dict[str, Dict[str, object]]],
    cache: Dict[tuple, Dict[str, List[object]]],
) -> List[OverlayEvent]:
    if not _camera_allowed(spec.camera, detect_camera):
        return []
    cfg = _resolve_vehicle_config(spec, detection_overrides)
    if spec.camera == "cabin":
        path = cabin
    elif spec.camera == "road":
        path = road
    else:
        path = road

    def _hashable_value(value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), _hashable_value(v)) for k, v in value.items()))
        return value

    cache_key = (
        str(path),
        tuple(sorted((k, _hashable_value(v)) for k, v in cfg.__dict__.items())),
    )
    if cache_key not in cache:
        cache[cache_key] = detect_vehicle_dynamics(video_path=path, config=cfg)

    kind_raw = str(spec.params.get("kind", "")).strip().lower()
    kind_map = {
        "moving": "vehicle_moving",
        "stopped": "vehicle_stopped",
        "accel": "vehicle_accel",
        "decel": "vehicle_decel",
        "turn": "vehicle_turn",
    }
    target_kind = kind_map.get(kind_raw, kind_raw)
    if target_kind not in cache[cache_key]:
        return []

    events = cache[cache_key][target_kind]
    scores = [float(getattr(e, "score", 0.0)) for e in events]
    strengths = _normalize_strength(scores)
    lane_events: List[OverlayEvent] = []
    for i, e in enumerate(events):
        start = float(getattr(e, "start", 0.0))
        end = float(getattr(e, "end", 0.0))
        meta = getattr(e, "meta", {}) or {}
        peak = float(meta.get("peak_s", (start + end) / 2.0))
        label = spec.title
        direction = meta.get("direction")
        if isinstance(direction, str) and direction:
            label = f"{label} {direction.upper()}"
        lane_events.append(
            OverlayEvent(
                start=start,
                end=end,
                peak=peak,
                label=label,
                strength01=strengths[i] if i < len(strengths) else 1.0,
            )
        )
    return _merge_overlapping(lane_events)


def build_conical_lanes(
    road: Path,
    cabin: Path,
    detect_camera: str = "both",
    max_duration: Optional[float] = None,
    config_path: Optional[Path] = None,
) -> List[Lane]:
    loader = ConfigLoader.from_env()
    if config_path is not None:
        config_path = Path(config_path).resolve()
        loader = ConfigLoader(repo_root=loader.repo_root, config_dir=config_path.parent)

    lanes_data = loader.load_lanes()
    lanes_path = loader.config_dir / "lanes.toml"
    legacy_path = loader.config_dir / "conicals.toml"
    legacy_config = (not lanes_path.exists()) and legacy_path.exists()
    specs = _load_lane_specs(lanes_data, legacy_config)

    scoring_models = _parse_scoring_models(loader.load_scoring())
    detection_overrides = _parse_detection_overrides(loader.load_detection_overrides())
    lane_keys = [spec.key for spec in specs if spec.detector != "score"]
    vehicle_cache: Dict[tuple, Dict[str, List[object]]] = {}
    registry: Dict[str, Callable[[LaneSpec], List[OverlayEvent]]] = {
        "motion": lambda spec: _build_motion_lane(
            spec, road, cabin, detect_camera, detection_overrides
        ),
        "audio": lambda spec: _build_audio_lane(spec, road, cabin, detect_camera),
        "optical_flow": lambda spec: _build_optical_flow_lane(
            spec, road, cabin, detect_camera, detection_overrides
        ),
        "vehicle_dynamics": lambda spec: _build_vehicle_lane(
            spec, road, cabin, detect_camera, detection_overrides, vehicle_cache
        ),
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
        model_name = spec.model or "weighted_union"
        model_cfg = scoring_models.get(model_name)
        if model_cfg is None:
            valid = ", ".join(sorted(scoring_models.keys()))
            raise ValueError(f"Unknown scoring model {model_name!r}. Valid: {valid}")
        _validate_score_model(model_name, model_cfg, lane_keys)
        built[spec.key] = _clip_lane(
            _build_score_lane(spec, built, model_cfg), max_duration
        )

    lanes: List[Lane] = []
    for spec in specs:
        if not spec.enabled:
            continue
        lanes.append(Lane(key=spec.key, title=spec.title, events=built.get(spec.key, [])))
    return lanes
