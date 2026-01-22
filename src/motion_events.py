"""
motion_events.py

Motion/activity detection for dashcam video.

This module computes motion scores and/or detects time ranges ("events") where
visual activity is elevated, suitable for selecting interesting driving moments.
It is designed for CPU-only execution and batch processing of many short clips.

Typical responsibilities:
- Sampling frames or downscaled video for lightweight motion estimation
- Producing per-time-bin motion intensity series
- Converting intensity peaks into event intervals with configurable thresholds
  and smoothing/debouncing

Outputs from this module are generally consumed by:
- selection/scoring logic (top-N moments)
- debug visualizations / captions
- layout/render orchestration to extract and stitch segments

This module should NOT:
- Assume a specific UI layout or caption format
- Invoke final rendering directly (layout handles that)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection_profiles import (
    MotionEventDetectConfig,
    OpticalFlowDetectConfig,
    VehicleDynamicsDetectConfig,
)

from .ingest import DashcamConfig, discover_pairs


@dataclass
class MotionEvent:
    """Represents a short segment around a motion spike."""
    start_time: float  # seconds
    peak_time: float   # seconds
    end_time: float    # seconds
    score: float       # motion score at peak

    def __str__(self) -> str:
        return (
            f"{self.start_time:6.2f}s → {self.end_time:6.2f}s "
            f"(peak {self.peak_time:6.2f}s, score={self.score:.3f})"
        )


@dataclass
class SmoothSegment:
    """A stretch of smooth, flowing motion (good driving aesthetic)."""
    start_time: float  # seconds
    end_time: float    # seconds
    mean_flow: float   # average optical flow magnitude over segment
    peak_flow: float | None = None  # peak flow magnitude over segment

    def __str__(self) -> str:
        peak = "" if self.peak_flow is None else f", peak_flow={self.peak_flow:.3f}"
        return (
            f"{self.start_time:6.2f}s → {self.end_time:6.2f}s "
            f"(mean_flow={self.mean_flow:.3f}{peak})"
        )


@dataclass
class VehicleEvent:
    start: float
    end: float
    kind: str
    score: float
    meta: Dict[str, object]


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _smooth_signal(signal: np.ndarray, mode: str, kernel: int, ema_alpha: float) -> np.ndarray:
    if mode == "none":
        return signal
    if mode == "ema":
        out = np.empty_like(signal)
        alpha = float(ema_alpha)
        out[0] = signal[0]
        for i in range(1, len(signal)):
            out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1]
        return out
    if mode == "median":
        k = _ensure_odd(int(kernel))
        pad = k // 2
        padded = np.pad(signal, (pad, pad), mode="edge")
        out = np.empty_like(signal)
        for i in range(len(signal)):
            out[i] = float(np.median(padded[i:i + k]))
        return out
    return signal


def _stat_value(arr: np.ndarray, mode: str) -> float:
    if arr.size == 0:
        return 0.0
    if mode == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _mad(arr: np.ndarray) -> float:
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _robust_threshold_global(signal: np.ndarray, k: float) -> float:
    med = float(np.median(signal))
    mad = _mad(signal) * 1.4826
    return med + k * mad


def _robust_threshold_rolling(signal: np.ndarray, window_samples: int, k: float) -> np.ndarray:
    w = max(1, int(window_samples))
    out = np.empty_like(signal)
    for i in range(len(signal)):
        start = max(0, i - w + 1)
        window = signal[start:i + 1]
        med = float(np.median(window))
        mad = _mad(window) * 1.4826
        out[i] = med + k * mad
    return out


def _apply_hysteresis(signal: np.ndarray, high_thr: np.ndarray | float, low_ratio: float) -> np.ndarray:
    if np.isscalar(high_thr):
        high = float(high_thr)
        low = high * float(low_ratio)
        active = False
        mask = np.zeros_like(signal, dtype=bool)
        for i, v in enumerate(signal):
            if not active and v >= high:
                active = True
            elif active and v <= low:
                active = False
            mask[i] = active
        return mask
    high = np.asarray(high_thr)
    low = high * float(low_ratio)
    active = False
    mask = np.zeros_like(signal, dtype=bool)
    for i, v in enumerate(signal):
        if not active and v >= high[i]:
            active = True
        elif active and v <= low[i]:
            active = False
        mask[i] = active
    return mask


def _find_runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _remove_short_runs(mask: np.ndarray, min_samples: int) -> np.ndarray:
    if min_samples <= 1:
        return mask
    out = mask.copy()
    for (s_idx, e_idx) in _find_runs(mask):
        if (e_idx - s_idx + 1) < min_samples:
            out[s_idx:e_idx + 1] = False
    return out


def _compute_optical_flow_stats(
    video_path: Path,
    downscale_width: int,
    frame_step: int,
    stat_fn: Callable[[np.ndarray], Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    prev_gray = None
    times: List[float] = []
    stats: Dict[str, List[float]] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        scale = downscale_width / float(w)
        new_size = (downscale_width, int(h * scale))
        frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            frame_idx += 1
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        stat_values = stat_fn(flow)

        t = frame_idx / fps
        times.append(t)
        for k, v in stat_values.items():
            stats.setdefault(k, []).append(float(v))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not times:
        return np.array([]), {}, float(fps), float(total_frames)

    times_arr = np.array(times)
    stats_arr = {k: np.array(v) for k, v in stats.items()}
    duration_s = (float(total_frames) / fps) if total_frames > 0 else float(times_arr[-1])
    return times_arr, stats_arr, float(fps), float(duration_s)


def detect_motion_events(
    video_path: Path,
    downscale_width: int = 320,
    frame_step: int = 2,
    pre_event_sec: float = 3.0,
    post_event_sec: float = 5.0,
    min_event_gap_sec: float = 5.0,
    threshold_std: float = 2.0,
    max_events: int = 20,
    config: MotionEventDetectConfig | dict | None = None,
) -> List[MotionEvent] | tuple[List[MotionEvent], dict]:
    """
    Basic motion-spike detector on a single video.

    - downscale_width: shrink frames for speed.
    - frame_step: analyze every Nth frame (2 => ~15fps if original is 30fps).
    - pre_event_sec/post_event_sec: segment padding around spike.
    - min_event_gap_sec: minimum separation between distinct events.
    - threshold_std: how many std dev above mean to call something a spike.
    - max_events: keep strongest N events.
    """
    base_cfg = MotionEventDetectConfig(
        downscale_width=downscale_width,
        frame_step=frame_step,
        pre_event_sec=pre_event_sec,
        post_event_sec=post_event_sec,
        min_event_gap_sec=min_event_gap_sec,
        threshold_std=threshold_std,
        max_events=max_events,
    )
    if config is None:
        cfg = base_cfg
    elif isinstance(config, MotionEventDetectConfig):
        cfg = MotionEventDetectConfig.from_dict(config.__dict__, base=base_cfg)
    else:
        cfg = MotionEventDetectConfig.from_dict(config, base=base_cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    print(f"[motion] Analyzing {video_path.name} at {fps:.2f} fps")

    prev_gray = None
    scores: List[float] = []
    times: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % cfg.frame_step != 0:
            frame_idx += 1
            continue

        # Downscale
        h, w = frame.shape[:2]
        scale = cfg.downscale_width / float(w)
        new_size = (cfg.downscale_width, int(h * scale))
        frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            frame_idx += 1
            continue

        diff = cv2.absdiff(gray, prev_gray)
        score = float(diff.mean())  # average absolute change per pixel

        t = frame_idx / fps
        scores.append(score)
        times.append(t)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not scores:
        return [] if not cfg.return_debug else ([], {"reason": "no_scores"})

    scores_arr = np.array(scores)
    times_arr = np.array(times)
    duration_s = (float(total_frames) / fps) if total_frames > 0 else float(times_arr[-1])
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0

    smooth = _smooth_signal(
        scores_arr,
        mode=cfg.smoothing_mode,
        kernel=cfg.smoothing_kernel,
        ema_alpha=cfg.ema_alpha,
    )

    if cfg.threshold_mode == "robust_global":
        high_thr = _robust_threshold_global(smooth, cfg.threshold_std)
    elif cfg.threshold_mode == "robust_rolling":
        window_samples = max(1, int(cfg.rolling_window_sec * (fps / cfg.frame_step)))
        high_thr = _robust_threshold_rolling(smooth, window_samples, cfg.threshold_std)
    else:
        mean = float(smooth.mean())
        std = float(smooth.std() or 1e-6)
        high_thr = mean + cfg.threshold_std * std
        print(
            f"[motion] mean={mean:.3f}, std={std:.3f}, "
            f"threshold={high_thr:.3f}"
        )

    if cfg.hysteresis:
        mask = _apply_hysteresis(smooth, high_thr, cfg.low_ratio)
    else:
        mask = smooth >= high_thr

    runs = _find_runs(mask)
    if not runs:
        print("[motion] No spikes above threshold.")
        return [] if not cfg.return_debug else ([], {"raw": scores_arr, "smooth": smooth, "thr": high_thr, "mask": mask})

    candidates: List[MotionEvent] = []
    for (s_idx, e_idx) in runs:
        run_scores = scores_arr[s_idx:e_idx + 1]
        peak_rel = int(np.argmax(run_scores))
        peak_idx = s_idx + peak_rel
        peak_score = float(scores_arr[peak_idx])
        peak_t = float(times_arr[peak_idx])
        start_t = max(0.0, peak_t - cfg.pre_event_sec)
        end_t = min(duration_s, peak_t + cfg.post_event_sec)
        candidates.append(
            MotionEvent(
                start_time=start_t,
                peak_time=peak_t,
                end_time=end_t,
                score=peak_score,
            )
        )

    # Merge overlapping or near-adjacent windows
    candidates.sort(key=lambda e: e.start_time)
    merged: List[MotionEvent] = []
    for ev in candidates:
        if not merged:
            merged.append(ev)
            continue
        last = merged[-1]
        if ev.start_time <= last.end_time + cfg.merge_gap_sec:
            best = ev if ev.score > last.score else last
            merged[-1] = MotionEvent(
                start_time=min(last.start_time, ev.start_time),
                peak_time=best.peak_time,
                end_time=max(last.end_time, ev.end_time),
                score=best.score,
            )
        else:
            merged.append(ev)

    # Keep strongest N, then sort chronologically
    merged_sorted = sorted(merged, key=lambda e: e.score, reverse=True)[: cfg.max_events]
    merged_sorted = sorted(merged_sorted, key=lambda e: e.start_time)

    # Final prune for compatibility with min_event_gap_sec
    pruned: List[MotionEvent] = []
    last_peak = -1e9
    for ev in merged_sorted:
        if ev.peak_time < last_peak + cfg.min_event_gap_sec:
            continue
        pruned.append(ev)
        last_peak = ev.peak_time

    print(f"[motion] Detected {len(pruned)} motion events.")
    if cfg.return_debug:
        # thr uses threshold_std as k for robust modes.
        debug = {
            "raw": scores_arr,
            "smooth": smooth,
            "thr": high_thr,
            "mask": mask,
            "times": times_arr,
        }
        return pruned, debug
    return pruned


def detect_optical_flow(
    video_path: Path,
    downscale_width: int = 320,
    frame_step: int = 2,
    min_duration_sec: float = 2.0,
    flow_percentile: float = 60.0,
    config: OpticalFlowDetectConfig | dict | None = None,
) -> List[SmoothSegment] | tuple[List[SmoothSegment], dict]:
    """
    Detect segments of sustained motion using only optical-flow magnitude.

    NOTE: This is experimental / unused by the CLI right now.

    Simpler heuristic:
    - Compute optical flow magnitude between frames.
    - Choose a threshold as the given percentile of magnitudes.
    - Any time range where flow >= threshold is considered "moving enough".
    - Group consecutive frames above threshold into segments of at least
      `min_duration_sec`.
    """
    base_cfg = OpticalFlowDetectConfig(
        downscale_width=downscale_width,
        frame_step=frame_step,
        min_duration_sec=min_duration_sec,
        flow_percentile=flow_percentile,
    )
    if config is None:
        cfg = base_cfg
    elif isinstance(config, OpticalFlowDetectConfig):
        cfg = OpticalFlowDetectConfig.from_dict(config.__dict__, base=base_cfg)
    else:
        cfg = OpticalFlowDetectConfig.from_dict(config, base=base_cfg)

    def stat_fn(flow: np.ndarray) -> Dict[str, float]:
        fx, fy = flow[..., 0], flow[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy)
        return {"mag_mean": float(mag.mean())}

    times_arr, stats, fps, duration_s = _compute_optical_flow_stats(
        video_path=video_path,
        downscale_width=cfg.downscale_width,
        frame_step=cfg.frame_step,
        stat_fn=stat_fn,
    )
    if times_arr.size == 0:
        print("[smooth] No frames processed.")
        return [] if not cfg.return_debug else ([], {"reason": "no_frames"})

    mag_arr = stats.get("mag_mean", np.array([]))
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0

    smooth = _smooth_signal(
        mag_arr,
        mode=cfg.smoothing_mode,
        kernel=cfg.smoothing_kernel,
        ema_alpha=cfg.ema_alpha,
    )

    if cfg.threshold_mode == "robust_global":
        thr = _robust_threshold_global(smooth, 1.0)
    elif cfg.threshold_mode == "robust_rolling":
        window_samples = max(1, int(cfg.rolling_window_sec * (fps / cfg.frame_step)))
        thr = _robust_threshold_rolling(smooth, window_samples, 1.0)
    else:
        thr = float(np.percentile(smooth, cfg.flow_percentile))

    if np.isscalar(thr):
        thr_eff = max(float(thr), cfg.abs_threshold)
    else:
        thr_eff = np.maximum(thr, cfg.abs_threshold)

    if cfg.hysteresis:
        moving_mask = _apply_hysteresis(smooth, thr_eff, cfg.low_ratio)
    else:
        moving_mask = smooth >= thr_eff

    runs = _find_runs(moving_mask)
    segments: List[SmoothSegment] = []
    for (s_idx, e_idx) in runs:
        start_t = float(times_arr[s_idx])
        end_t = float(times_arr[e_idx])
        run_duration = (end_t - start_t) + dt_sec
        if run_duration < cfg.min_duration_sec:
            continue
        if cfg.pad_sec > 0:
            start_t = max(0.0, start_t - cfg.pad_sec)
            end_t = min(duration_s, end_t + cfg.pad_sec)
        segments.append(
            SmoothSegment(
                start_time=start_t,
                end_time=end_t,
                mean_flow=0.0,
                peak_flow=0.0,
            )
        )

    # Merge segments with small gaps
    segments.sort(key=lambda s: s.start_time)
    merged: List[SmoothSegment] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg.start_time <= last.end_time + cfg.merge_gap_sec:
            merged[-1] = SmoothSegment(
                start_time=max(0.0, min(last.start_time, seg.start_time)),
                end_time=min(duration_s, max(last.end_time, seg.end_time)),
                mean_flow=0.0,
                peak_flow=0.0,
            )
        else:
            merged.append(
                SmoothSegment(
                    start_time=max(0.0, seg.start_time),
                    end_time=min(duration_s, seg.end_time),
                    mean_flow=0.0,
                    peak_flow=0.0,
                )
            )

    # Recompute stats on merged spans
    final_segments: List[SmoothSegment] = []
    for seg in merged:
        start_t = max(0.0, seg.start_time)
        end_t = min(duration_s, seg.end_time)
        idx = np.where((times_arr >= start_t) & (times_arr <= end_t))[0]
        if idx.size == 0:
            mean_flow = 0.0
            peak_flow = 0.0
        else:
            mean_flow = float(mag_arr[idx].mean())
            peak_flow = float(mag_arr[idx].max())
        final_segments.append(
            SmoothSegment(
                start_time=start_t,
                end_time=end_t,
                mean_flow=mean_flow,
                peak_flow=peak_flow,
            )
        )

    print(f"[smooth] Detected {len(final_segments)} smooth segments.")
    if cfg.return_debug:
        # thr uses robust median+MAD (k=1.0) or percentile depending on mode.
        debug = {
            "raw": mag_arr,
            "smooth": smooth,
            "thr": thr_eff,
            "mask": moving_mask,
            "times": times_arr,
        }
        return final_segments, debug
    return final_segments


# ---------------------------------------------------------------------------


def _resolve_roi_bounds(
    roi: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    if max(x0, y0, x1, y1) <= 1.0:
        x0 = int(round(x0 * width))
        x1 = int(round(x1 * width))
        y0 = int(round(y0 * height))
        y1 = int(round(y1 * height))
    x0 = max(0, min(width - 1, int(round(x0))))
    x1 = max(1, min(width, int(round(x1))))
    y0 = max(0, min(height - 1, int(round(y0))))
    y1 = max(1, min(height, int(round(y1))))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return 0, 0, width, height
    return x0, y0, x1, y1


def _compute_vehicle_flow_series(
    video_path: Path,
    cfg: VehicleDynamicsDetectConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    roi_bounds = {"set": False, "roi": (0, 0, 0, 0)}

    def stat_fn(flow: np.ndarray) -> Dict[str, float]:
        fx, fy = flow[..., 0], flow[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy)
        h, w = mag.shape[:2]
        if not roi_bounds["set"]:
            roi_bounds["roi"] = _resolve_roi_bounds(cfg.road_roi, w, h)
            roi_bounds["set"] = True
        x0, y0, x1, y1 = roi_bounds["roi"]
        mag_roi = mag[y0:y1, x0:x1]
        fx_roi = fx[y0:y1, x0:x1]
        if mag_roi.size == 0 or fx_roi.size == 0:
            speed_proxy = _stat_value(mag, cfg.flow_stat)
            yaw_proxy = 0.0
        else:
            speed_proxy = _stat_value(mag_roi, cfg.flow_stat)
            mid = x0 + (x1 - x0) // 2
            fx_left = fx[y0:y1, x0:mid]
            fx_right = fx[y0:y1, mid:x1]
            if fx_left.size == 0 or fx_right.size == 0:
                yaw_proxy = 0.0
            else:
                yaw_proxy = _stat_value(fx_right, cfg.flow_stat) - _stat_value(
                    fx_left, cfg.flow_stat
                )
        return {"speed": speed_proxy, "yaw": yaw_proxy}

    times_arr, stats, fps, duration_s = _compute_optical_flow_stats(
        video_path=video_path,
        downscale_width=cfg.downscale_width,
        frame_step=cfg.frame_step,
        stat_fn=stat_fn,
    )
    if times_arr.size == 0:
        return np.array([]), np.array([]), np.array([]), 0.0
    speed = stats.get("speed", np.array([]))
    yaw = stats.get("yaw", np.array([]))
    dt_sec = (float(cfg.frame_step) / fps) if fps and np.isfinite(fps) else 0.0
    return times_arr, speed, yaw, dt_sec


def _smooth_seconds(signal: np.ndarray, seconds: float, dt_sec: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    if seconds <= 0 or dt_sec <= 0:
        return signal
    kernel = max(1, int(round(seconds / dt_sec)))
    kernel = _ensure_odd(kernel)
    return _smooth_signal(signal, mode="median", kernel=kernel, ema_alpha=0.0)


def detect_vehicle_state_segments(
    times: np.ndarray,
    speed_proxy: np.ndarray,
    cfg: VehicleDynamicsDetectConfig,
    duration_s: Optional[float] = None,
) -> Tuple[List[VehicleEvent], List[VehicleEvent]]:
    if times.size == 0 or speed_proxy.size == 0:
        return [], []
    dt_sec = float(times[1] - times[0]) if times.size > 1 else 0.0
    smooth = _smooth_seconds(speed_proxy, cfg.speed_smooth_seconds, dt_sec)
    high_thr = float(cfg.moving_threshold)
    if cfg.stopped_threshold is not None and high_thr > 0:
        low_ratio = float(cfg.stopped_threshold) / high_thr
    else:
        low_ratio = 0.8
    moving_mask = _apply_hysteresis(smooth, high_thr, low_ratio)
    min_samples = max(1, int(round(cfg.state_hysteresis_seconds / dt_sec))) if dt_sec > 0 else 1
    moving_mask = _remove_short_runs(moving_mask, min_samples)
    stopped_mask = _remove_short_runs(~moving_mask, min_samples)

    duration = float(duration_s) if duration_s is not None else float(times[-1])
    moving_events: List[VehicleEvent] = []
    for (s_idx, e_idx) in _find_runs(moving_mask):
        start_t = float(times[s_idx])
        end_t = float(times[e_idx] + dt_sec)
        mean_speed = float(np.mean(smooth[s_idx:e_idx + 1]))
        moving_events.append(
            VehicleEvent(
                start=start_t,
                end=min(duration, end_t),
                kind="vehicle_moving",
                score=mean_speed,
                meta={"peak_s": (start_t + end_t) / 2.0},
            )
        )

    stopped_events: List[VehicleEvent] = []
    for (s_idx, e_idx) in _find_runs(stopped_mask):
        start_t = float(times[s_idx])
        end_t = float(times[e_idx] + dt_sec)
        mean_speed = float(np.mean(smooth[s_idx:e_idx + 1]))
        score = max(0.0, high_thr - mean_speed)
        stopped_events.append(
            VehicleEvent(
                start=start_t,
                end=min(duration, end_t),
                kind="vehicle_stopped",
                score=score,
                meta={"peak_s": (start_t + end_t) / 2.0},
            )
        )
    return moving_events, stopped_events


def _find_peak_indices(
    signal: np.ndarray,
    mask: np.ndarray,
    min_separation_samples: int,
    peak_mode: str,
) -> List[int]:
    runs = _find_runs(mask)
    peaks: List[int] = []
    for (s_idx, e_idx) in runs:
        window = signal[s_idx:e_idx + 1]
        if window.size == 0:
            continue
        if peak_mode == "max":
            peak_rel = int(np.argmax(window))
        else:
            peak_rel = int(np.argmin(window))
        peaks.append(s_idx + peak_rel)

    if not peaks:
        return []

    peaks = sorted(peaks)
    filtered: List[int] = []
    for idx in peaks:
        if not filtered:
            filtered.append(idx)
            continue
        last = filtered[-1]
        if abs(idx - last) < min_separation_samples:
            choose_idx = idx if abs(signal[idx]) > abs(signal[last]) else last
            filtered[-1] = choose_idx
        else:
            filtered.append(idx)
    return filtered


def _detect_spikes(
    times: np.ndarray,
    series: np.ndarray,
    threshold: Optional[float],
    zscore_threshold: Optional[float],
    min_separation_sec: float,
    pad_sec: float,
    kind: str,
    duration_s: Optional[float],
    peak_mode: str,
) -> List[VehicleEvent]:
    if times.size == 0 or series.size == 0:
        return []
    dt_sec = float(times[1] - times[0]) if times.size > 1 else 0.0
    if threshold is None and zscore_threshold is not None:
        med = float(np.median(series))
        mad = _mad(series) * 1.4826
        if mad <= 1e-9:
            threshold = med + (zscore_threshold * (1.0 if peak_mode == "max" else -1.0))
        else:
            threshold = med + (zscore_threshold * mad * (1.0 if peak_mode == "max" else -1.0))

    if threshold is None:
        return []

    if peak_mode == "max":
        mask = series >= float(threshold)
    else:
        mask = series <= float(threshold)

    min_sep_samples = max(1, int(round(min_separation_sec / dt_sec))) if dt_sec > 0 else 1
    peak_indices = _find_peak_indices(series, mask, min_sep_samples, peak_mode=peak_mode)
    duration = float(duration_s) if duration_s is not None else float(times[-1])
    events: List[VehicleEvent] = []
    for idx in peak_indices:
        peak_t = float(times[idx])
        start_t = max(0.0, peak_t - pad_sec)
        end_t = min(duration, peak_t + pad_sec)
        score = float(abs(series[idx]))
        events.append(
            VehicleEvent(
                start=start_t,
                end=end_t,
                kind=kind,
                score=score,
                meta={"peak_s": peak_t, "value": float(series[idx])},
            )
        )
    return events


def detect_accel_spikes(
    times: np.ndarray,
    speed_proxy: np.ndarray,
    cfg: VehicleDynamicsDetectConfig,
    duration_s: Optional[float] = None,
) -> List[VehicleEvent]:
    if times.size < 2:
        return []
    dt_sec = float(times[1] - times[0]) if times.size > 1 else 0.0
    smooth_speed = _smooth_seconds(speed_proxy, cfg.speed_smooth_seconds, dt_sec)
    accel = np.gradient(smooth_speed, times)
    accel = _smooth_seconds(accel, cfg.accel_smooth_seconds, dt_sec)
    return _detect_spikes(
        times=times,
        series=accel,
        threshold=cfg.accel_threshold,
        zscore_threshold=cfg.accel_zscore_threshold,
        min_separation_sec=cfg.spike_min_separation_seconds,
        pad_sec=cfg.spike_event_padding_seconds,
        kind="vehicle_accel",
        duration_s=duration_s,
        peak_mode="max",
    )


def detect_decel_spikes(
    times: np.ndarray,
    speed_proxy: np.ndarray,
    cfg: VehicleDynamicsDetectConfig,
    duration_s: Optional[float] = None,
) -> List[VehicleEvent]:
    if times.size < 2:
        return []
    dt_sec = float(times[1] - times[0]) if times.size > 1 else 0.0
    smooth_speed = _smooth_seconds(speed_proxy, cfg.speed_smooth_seconds, dt_sec)
    accel = np.gradient(smooth_speed, times)
    accel = _smooth_seconds(accel, cfg.accel_smooth_seconds, dt_sec)
    return _detect_spikes(
        times=times,
        series=accel,
        threshold=cfg.decel_threshold,
        zscore_threshold=cfg.decel_zscore_threshold,
        min_separation_sec=cfg.spike_min_separation_seconds,
        pad_sec=cfg.spike_event_padding_seconds,
        kind="vehicle_decel",
        duration_s=duration_s,
        peak_mode="min",
    )


def detect_turn_segments(
    times: np.ndarray,
    yaw_proxy: np.ndarray,
    cfg: VehicleDynamicsDetectConfig,
    duration_s: Optional[float] = None,
) -> List[VehicleEvent]:
    if times.size == 0 or yaw_proxy.size == 0:
        return []
    dt_sec = float(times[1] - times[0]) if times.size > 1 else 0.0
    yaw_smooth = _smooth_seconds(yaw_proxy, cfg.yaw_smooth_seconds, dt_sec)
    mask = np.abs(yaw_smooth) >= float(cfg.turn_threshold)
    runs = _find_runs(mask)
    duration = float(duration_s) if duration_s is not None else float(times[-1])
    segments: List[Tuple[int, int]] = []
    min_samples = max(1, int(round(cfg.turn_min_duration_seconds / dt_sec))) if dt_sec > 0 else 1
    for (s_idx, e_idx) in runs:
        if (e_idx - s_idx + 1) >= min_samples:
            segments.append((s_idx, e_idx))

    # merge gaps
    merged: List[Tuple[int, int]] = []
    max_gap = int(round(cfg.turn_merge_gap_seconds / dt_sec)) if dt_sec > 0 else 0
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        last_s, last_e = merged[-1]
        if seg[0] <= last_e + max_gap:
            merged[-1] = (last_s, max(last_e, seg[1]))
        else:
            merged.append(seg)

    events: List[VehicleEvent] = []
    for (s_idx, e_idx) in merged:
        start_t = float(times[s_idx])
        end_t = float(times[e_idx] + dt_sec)
        segment = yaw_smooth[s_idx:e_idx + 1]
        direction = "right" if float(np.median(segment)) > 0 else "left"
        score = float(np.median(np.abs(segment)))
        events.append(
            VehicleEvent(
                start=start_t,
                end=min(duration, end_t),
                kind=f"vehicle_turn_{direction}",
                score=score,
                meta={"peak_s": (start_t + end_t) / 2.0, "direction": direction},
            )
        )
    return events


def detect_vehicle_dynamics(
    video_path: Path,
    config: VehicleDynamicsDetectConfig | dict | None = None,
) -> Dict[str, List[VehicleEvent]]:
    base_cfg = VehicleDynamicsDetectConfig()
    if config is None:
        cfg = base_cfg
    elif isinstance(config, VehicleDynamicsDetectConfig):
        cfg = VehicleDynamicsDetectConfig.from_dict(config.__dict__, base=base_cfg)
    else:
        cfg = VehicleDynamicsDetectConfig.from_dict(config, base=base_cfg)

    times, speed_proxy, yaw_proxy, dt_sec = _compute_vehicle_flow_series(
        video_path=video_path,
        cfg=cfg,
    )
    if times.size == 0:
        empty = {
            "vehicle_moving": [],
            "vehicle_stopped": [],
            "vehicle_accel": [],
            "vehicle_decel": [],
            "vehicle_turn": [],
        }
        return empty

    duration_s = float(times[-1] + dt_sec)
    moving, stopped = detect_vehicle_state_segments(
        times, speed_proxy, cfg, duration_s=duration_s
    )
    accel = detect_accel_spikes(times, speed_proxy, cfg, duration_s=duration_s)
    decel = detect_decel_spikes(times, speed_proxy, cfg, duration_s=duration_s)
    turns = detect_turn_segments(times, yaw_proxy, cfg, duration_s=duration_s)

    by_kind: Dict[str, List[VehicleEvent]] = {
        "vehicle_moving": moving,
        "vehicle_stopped": stopped,
        "vehicle_accel": accel,
        "vehicle_decel": decel,
        "vehicle_turn": turns,
    }
    return by_kind



def _select_video_from_pair(pair, camera: str) -> Path:
    """
    Utility to pick the right video from a road/cabin pair.
    camera: 'road' | 'cabin'
    """
    if camera == "road":
        return pair.road
    elif camera == "cabin":
        return pair.cabin
    else:
        raise ValueError(f"Unknown camera: {camera!r} (expected 'road' or 'cabin')")


def demo_motion_on_pair(
    base_dir: Path,
    pair_index: int = 0,
    camera: str = "road",
) -> None:
    """
    Convenience function:
    - load road/cabin pair using ingest.py
    - run motion detection on the chosen camera
    - print out candidate event segments

    camera: 'road', 'cabin', or 'both'
    """
    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)

    if not pairs:
        raise RuntimeError("No clip pairs found.")
    if pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    print(f"Using pair #{pair_index}:")
    print("  road :", pair.road)
    print("  cabin:", pair.cabin)

    # If the user wants both, just run twice with labels.
    if camera == "both":
        cameras_to_run: List[str] = ["road", "cabin"]
    else:
        cameras_to_run = [camera]

    for cam in cameras_to_run:
        video_path = _select_video_from_pair(pair, cam)
        print(f"\n[motion] Running on {cam.upper()} camera ({video_path.name})")
        events = detect_motion_events(video_path)
        if not events:
            print("  No events detected.")
            continue

        print("\n  Candidate motion events:")
        for e in events:
            print("   ", e)


def demo_smooth_on_pair(
    base_dir: Path,
    pair_index: int = 0,
    camera: str = "road",
) -> None:
    """
    Run smooth segment detection on a chosen camera in a pair.

    NOTE: Not wired to the CLI right now; for experimentation only.

    camera: 'road', 'cabin', or 'both'
    """
    cfg = DashcamConfig(base_dir=base_dir)
    pairs = discover_pairs(cfg)

    if not pairs:
        raise RuntimeError("No clip pairs found.")
    if pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    print(f"Using pair #{pair_index}:")
    print("  road :", pair.road)
    print("  cabin:", pair.cabin)

    if camera == "both":
        cameras_to_run: List[str] = ["road", "cabin"]
    else:
        cameras_to_run = [camera]

    for cam in cameras_to_run:
        video_path = _select_video_from_pair(pair, cam)
        print(f"\n[smooth] Running on {cam.upper()} camera ({video_path.name})")
        segments = detect_optical_flow(video_path)
        if not segments:
            print("  No smooth segments detected.")
            continue

        print("\n  Smooth driving segments:")
        for s in segments:
            print("   ", s)

