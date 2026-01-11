from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

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

    def __str__(self) -> str:
        return (
            f"{self.start_time:6.2f}s → {self.end_time:6.2f}s "
            f"(mean_flow={self.mean_flow:.3f})"
        )


def detect_motion_events(
    video_path: Path,
    downscale_width: int = 320,
    frame_step: int = 2,
    pre_event_sec: float = 3.0,
    post_event_sec: float = 5.0,
    min_event_gap_sec: float = 5.0,
    threshold_std: float = 2.0,
    max_events: int = 20,
) -> List[MotionEvent]:
    """
    Basic motion-spike detector on a single video.

    - downscale_width: shrink frames for speed.
    - frame_step: analyze every Nth frame (2 => ~15fps if original is 30fps).
    - pre_event_sec/post_event_sec: segment padding around spike.
    - min_event_gap_sec: minimum separation between distinct events.
    - threshold_std: how many std dev above mean to call something a spike.
    - max_events: keep strongest N events.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[motion] Analyzing {video_path.name} at {fps:.2f} fps")

    prev_gray = None
    scores: List[float] = []
    times: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # Downscale
        h, w = frame.shape[:2]
        scale = downscale_width / float(w)
        new_size = (downscale_width, int(h * scale))
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
        return []

    scores_arr = np.array(scores)
    times_arr = np.array(times)

    mean = scores_arr.mean()
    std = scores_arr.std() or 1e-6
    threshold = mean + threshold_std * std

    print(
        f"[motion] mean={mean:.3f}, std={std:.3f}, "
        f"threshold={threshold:.3f}"
    )

    # Indices where motion score exceeds threshold
    spike_indices = np.where(scores_arr > threshold)[0]
    if spike_indices.size == 0:
        print("[motion] No spikes above threshold.")
        return []

    events: List[MotionEvent] = []
    last_event_end = -1e9

    for idx in spike_indices:
        peak_t = float(times_arr[idx])
        if peak_t < last_event_end + min_event_gap_sec:
            # too close to previous event
            continue

        start_t = max(0.0, peak_t - pre_event_sec)
        end_t = peak_t + post_event_sec

        events.append(
            MotionEvent(
                start_time=start_t,
                peak_time=peak_t,
                end_time=end_t,
                score=float(scores_arr[idx]),
            )
        )
        last_event_end = end_t

    # If there are too many events, keep the strongest ones
    events_sorted = sorted(events, key=lambda e: e.score, reverse=True)
    events_sorted = events_sorted[:max_events]

    # Sort chronologically for readability
    events_sorted = sorted(events_sorted, key=lambda e: e.start_time)

    print(f"[motion] Detected {len(events_sorted)} motion events.")
    return events_sorted


def detect_smooth_segments(
    video_path: Path,
    downscale_width: int = 320,
    frame_step: int = 2,
    min_duration_sec: float = 2.0,
    flow_percentile: float = 60.0,
) -> List[SmoothSegment]:
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    prev_gray = None
    times: List[float] = []
    mag_means: List[float] = []

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
        fx, fy = flow[..., 0], flow[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy)
        mag_mean = float(mag.mean())

        t = frame_idx / fps
        times.append(t)
        mag_means.append(mag_mean)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not mag_means:
        print("[smooth] No frames processed.")
        return []

    times_arr = np.array(times)
    mag_arr = np.array(mag_means)

    thr = np.percentile(mag_arr, flow_percentile)
    print(
        f"[smooth] mean_flow={mag_arr.mean():.3f}, "
        f"threshold(p{flow_percentile})={thr:.3f}"
    )

    moving_mask = mag_arr >= thr

    segments: List[SmoothSegment] = []
    start_idx = None

    def flush_segment(end_idx: int):
        nonlocal start_idx
        if start_idx is None:
            return
        start_t = float(times_arr[start_idx])
        end_t = float(times_arr[end_idx])
        if end_t - start_t >= min_duration_sec:
            mean_flow = float(mag_arr[start_idx:end_idx + 1].mean())
            segments.append(
                SmoothSegment(
                    start_time=start_t,
                    end_time=end_t,
                    mean_flow=mean_flow,
                )
            )
        start_idx = None

    for i, is_moving in enumerate(moving_mask):
        if is_moving:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                flush_segment(i - 1)

    if start_idx is not None:
        flush_segment(len(moving_mask) - 1)

    print(f"[smooth] Detected {len(segments)} smooth segments.")
    return segments


# ---------------------------------------------------------------------------


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
        segments = detect_smooth_segments(video_path)
        if not segments:
            print("  No smooth segments detected.")
            continue

        print("\n  Smooth driving segments:")
        for s in segments:
            print("   ", s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect driving motion spikes on dashcam pairs."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to DCIM folder containing DCIMA / DCIMB...",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Clip pair index to analyze (0-based)",
    )
    parser.add_argument(
        "--mode",
        choices=["motion"],
        default="motion",
        help="What to run. (Currently only motion spikes are supported.)",
    )
    parser.add_argument(
        "--camera",
        choices=["road", "cabin", "both"],
        default="both",
        help="Which camera to analyze.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)

    # For now, we always run motion spikes; --mode exists for future extension.
    print("\n=== MOTION SPIKES ===")
    demo_motion_on_pair(base_dir=base, pair_index=args.index, camera=args.camera)
