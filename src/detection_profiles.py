"""
Detection config objects and presets.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, Tuple


@dataclass
class MotionEventDetectConfig:
    downscale_width: int = 320
    frame_step: int = 2
    pre_event_sec: float = 3.0
    post_event_sec: float = 5.0
    min_event_gap_sec: float = 1.0
    # threshold_std meaning:
    # - meanstd_global: threshold = mean + threshold_std * std
    # - robust_global/robust_rolling: threshold = median + threshold_std * (1.4826 * MAD)
    # threshold_std acts as a sensitivity factor ("k") in robust modes.
    threshold_std: float = 5.0
    max_events: int = 20
    smoothing_mode: str = "median"  # none | median | ema
    smoothing_kernel: int = 5
    ema_alpha: float = 0.2
    threshold_mode: str = "robust_global"  # meanstd_global | robust_global | robust_rolling
    rolling_window_sec: float = 20.0
    hysteresis: bool = True
    low_ratio: float = 0.6
    merge_gap_sec: float = 0.25
    return_debug: bool = False

    @classmethod
    def from_dict(
        cls, data: dict, base: MotionEventDetectConfig | None = None
    ) -> MotionEventDetectConfig:
        base = base or cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        merged = {**base.__dict__, **{k: v for k, v in data.items() if k in fields}}
        return cls(**merged)


@dataclass
class OpticalFlowDetectConfig:
    downscale_width: int = 320
    frame_step: int = 2
    min_duration_sec: float = 1.5
    flow_percentile: float = 80.0
    smoothing_mode: str = "median"  # none | median | ema
    smoothing_kernel: int = 5
    ema_alpha: float = 0.2
    threshold_mode: str = "percentile"  # percentile | robust_global | robust_rolling
    # robust_*: median + (1.4826 * MAD)  # implicit k=1.0
    rolling_window_sec: float = 20.0
    abs_threshold: float = 0.01  # scale-dependent flow-magnitude floor
    hysteresis: bool = True
    low_ratio: float = 0.6
    pad_sec: float = 0.5
    merge_gap_sec: float = 0.25
    return_debug: bool = False

    @classmethod
    def from_dict(
        cls, data: dict, base: OpticalFlowDetectConfig | None = None
    ) -> OpticalFlowDetectConfig:
        base = base or cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        merged = {**base.__dict__, **{k: v for k, v in data.items() if k in fields}}
        return cls(**merged)


@dataclass
class VehicleDynamicsDetectConfig:
    # ROI is (x0, y0, x1, y1) in normalized or pixel coordinates.
    downscale_width: int = 320
    frame_step: int = 2
    road_roi: Tuple[float, float, float, float] = (0.2, 0.55, 0.8, 0.92)
    flow_stat: str = "median"
    speed_smooth_seconds: float = 5.0
    state_hysteresis_seconds: float = 3.0
    moving_threshold: float = 0.20
    stopped_threshold: Optional[float] = 0.12

    # Accel / decel spikes
    accel_smooth_seconds: float = 2.0
    accel_threshold: Optional[float] = None
    accel_zscore_threshold: Optional[float] = 3.0
    decel_threshold: Optional[float] = None
    decel_zscore_threshold: Optional[float] = 3.0
    spike_min_separation_seconds: float = 2.5
    spike_event_padding_seconds: float = 1.0

    # Turning
    yaw_smooth_seconds: float = 2.0
    turn_threshold: float = 0.10
    turn_min_duration_seconds: float = 1.0
    turn_merge_gap_seconds: float = 0.5

    @classmethod
    def from_dict(
        cls, data: dict, base: VehicleDynamicsDetectConfig | None = None
    ) -> VehicleDynamicsDetectConfig:
        base = base or cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        merged = {**base.__dict__, **{k: v for k, v in data.items() if k in fields}}
        return cls(**merged)


MOTION_PROFILES: dict[str, MotionEventDetectConfig] = {
    "default": MotionEventDetectConfig(),
    "legacy_meanstd": MotionEventDetectConfig(
        threshold_mode="meanstd_global",
        smoothing_mode="none",
        hysteresis=False,
        threshold_std=2.0,
    ),
}

FLOW_PROFILES: dict[str, OpticalFlowDetectConfig] = {
    "default": OpticalFlowDetectConfig(),
    "legacy_percentile": OpticalFlowDetectConfig(
        flow_percentile=60.0,
        min_duration_sec=2.0,
    ),
}

VEHICLE_PROFILES: dict[str, VehicleDynamicsDetectConfig] = {
    "default": VehicleDynamicsDetectConfig(),
}


def get_motion_profile(name: str) -> MotionEventDetectConfig:
    try:
        profile = MOTION_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MOTION_PROFILES.keys()))
        raise ValueError(f"Unknown motion profile {name!r}. Valid names: {valid}") from exc
    return MotionEventDetectConfig.from_dict(profile.__dict__)


def get_flow_profile(name: str) -> OpticalFlowDetectConfig:
    try:
        profile = FLOW_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FLOW_PROFILES.keys()))
        raise ValueError(f"Unknown flow profile {name!r}. Valid names: {valid}") from exc
    return OpticalFlowDetectConfig.from_dict(profile.__dict__)


def get_vehicle_profile(name: str) -> VehicleDynamicsDetectConfig:
    try:
        profile = VEHICLE_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(VEHICLE_PROFILES.keys()))
        raise ValueError(f"Unknown vehicle profile {name!r}. Valid names: {valid}") from exc
    return VehicleDynamicsDetectConfig.from_dict(profile.__dict__)
