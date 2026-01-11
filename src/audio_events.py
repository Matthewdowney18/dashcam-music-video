from __future__ import annotations

import argparse
import dataclasses
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .ingest import DashcamConfig, discover_pairs


# ==========
# Data model
# ==========

@dataclasses.dataclass
class AudioEvent:
    """Represents a short segment around an audio spike / salient sound."""
    start: float
    end: float
    peak_time: float
    peak_score: float

    def __str__(self) -> str:
        return (
            f"{self.start:6.2f}s → {self.end:6.2f}s "
            f"(peak {self.peak_time:6.2f}s, score={self.peak_score:.3f})"
        )


# ==================
# Pair / camera util
# ==================

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


# =====================
# Core audio processing
# =====================

def extract_mono_audio(
    video_path: Path,
    sample_rate: int = 16_000,
) -> Tuple[np.ndarray, int]:
    """
    Extract mono float32 PCM audio from a video using ffmpeg.
    Returns (audio, sample_rate).
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}:\n"
            f"{e.stderr.decode(errors='ignore')}"
        ) from e

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"No audio data extracted from {video_path}")

    return audio, sample_rate


def frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    Simple strided framing. Pads the end with zeros if needed.
    Returns shape (num_frames, frame_size).
    """
    if x.ndim != 1:
        raise ValueError("Expected mono signal")

    n = x.shape[0]
    if n <= frame_size:
        pad = frame_size - n
        x_padded = np.pad(x, (0, pad), mode="constant")
        return x_padded[None, :]

    num_frames = 1 + (n - frame_size) // hop_size
    remainder = n - (num_frames - 1) * hop_size - frame_size
    if remainder > 0:
        pad = hop_size - remainder
        x = np.pad(x, (0, pad), mode="constant")
        n = x.shape[0]
        num_frames = 1 + (n - frame_size) // hop_size

    strides = (hop_size * x.strides[0], x.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(num_frames, frame_size), strides=strides
    )
    return frames.copy()


def compute_audio_features(
    audio: np.ndarray,
    sample_rate: int,
    win_sec: float = 0.05,
    hop_sec: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame audio features:
    - rms
    - spectral_flux
    - band_energy_low / mid / high (as ratios)
    - spectral centroid
    Returns dict with features and frame_times.
    """
    frame_size = int(win_sec * sample_rate)
    hop_size = int(hop_sec * sample_rate)
    frames = frame_signal(audio, frame_size, hop_size)

    # Hann window, energy-normalized
    window = np.hanning(frame_size).astype(np.float32)
    window /= np.sqrt((window**2).mean() + 1e-8)
    frames_win = frames * window[None, :]

    # RMS
    rms = np.sqrt(np.mean(frames_win**2, axis=1) + 1e-12)

    # FFT magnitude
    fft = np.fft.rfft(frames_win, axis=1)
    mag = np.abs(fft) + 1e-9

    # Frequency bins
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    # Band masks
    low_mask = freqs < 300.0
    mid_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    high_mask = freqs > 3400.0

    total_energy = (mag**2).sum(axis=1) + 1e-9
    low_energy = (mag[:, low_mask] ** 2).sum(axis=1)
    mid_energy = (mag[:, mid_mask] ** 2).sum(axis=1)
    high_energy = (mag[:, high_mask] ** 2).sum(axis=1)

    low_ratio = low_energy / total_energy
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    centroid = (mag * freqs[None, :]).sum(axis=1) / (mag.sum(axis=1) + 1e-9)

    # Spectral flux
    mag_norm = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
    flux = np.zeros(mag.shape[0], dtype=np.float32)
    flux[1:] = np.sqrt(((mag_norm[1:] - mag_norm[:-1]) ** 2).sum(axis=1))

    num_frames = frames.shape[0]
    times = np.arange(num_frames) * hop_sec

    return {
        "rms": rms,
        "flux": flux,
        "low_ratio": low_ratio,
        "mid_ratio": mid_ratio,
        "high_ratio": high_ratio,
        "centroid": centroid,
        "frame_times": times,
        "frame_hop_sec": hop_sec,
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - m) / s


def compute_event_scores(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine features into a single salience score.

    Two components:
    - transient_score: loud, spiky, often high-frequency (honks, crashes, wind)
    - voice_score: mid-band emphasis with moderate loudness (speech, yelling, music)

    Final score = max(transient_score, voice_score).
    """
    rms_z = _zscore(features["rms"])
    flux_z = _zscore(features["flux"])
    low_z = _zscore(features["low_ratio"])
    mid_z = _zscore(features["mid_ratio"])
    high_z = _zscore(features["high_ratio"])
    centroid_z = _zscore(features["centroid"])

    transient_score = (
        1.0 * rms_z
        + 0.7 * flux_z
        + 0.4 * high_z
        + 0.2 * centroid_z
    )

    voice_score = (
        0.5 * rms_z
        + 0.6 * mid_z
        - 0.2 * low_z
        - 0.1 * high_z
    )

    return np.maximum(transient_score, voice_score)


def detect_audio_events_from_scores(
    scores: np.ndarray,
    frame_times: np.ndarray,
    clip_duration: float,
    k: float = 1.8,
    pre_event_sec: float = 1.5,
    post_event_sec: float = 2.5,
    min_event_gap_sec: float = 1.0,
    max_events: int = 10,
) -> List[AudioEvent]:
    """
    Turn a per-frame score sequence into short audio events.

    New strategy:
      - find local maxima (peaks) above a dynamic threshold
      - take the top-N peaks by score
      - create fixed windows around each peak
      - enforce a minimum gap between peaks

    This avoids "one giant event covering the whole clip".
    """
    n = scores.size
    if n < 3:
        return []

    # Ignore NaNs / infs when computing stats
    valid = np.isfinite(scores)
    if not np.any(valid):
        return []

    scores_valid = scores[valid]
    mu = float(np.mean(scores_valid))
    sigma = float(np.std(scores_valid))
    if sigma < 1e-6:
        return []

    thr = mu + k * sigma

    # --- 1) find local maxima above threshold ---
    peak_indices: List[int] = []
    for i in range(1, n - 1):
        s = scores[i]
        if (
            s >= thr
            and s > scores[i - 1]
            and s >= scores[i + 1]
        ):
            peak_indices.append(i)

    if not peak_indices:
        return []

    peak_indices = np.asarray(peak_indices, dtype=int)

    # --- 2) sort peaks by score (highest first) ---
    sort_order = np.argsort(scores[peak_indices])[::-1]
    peak_indices = peak_indices[sort_order]

    # --- 3) pick peaks with a minimum gap between them ---
    events: List[AudioEvent] = []

    for idx in peak_indices:
        peak_time = float(frame_times[idx])

        # Enforce min gap between peak times
        too_close = False
        for ev in events:
            if abs(peak_time - ev.peak_time) < min_event_gap_sec:
                too_close = True
                break
        if too_close:
            continue

        start_time = max(0.0, peak_time - pre_event_sec)
        end_time = min(clip_duration, peak_time + post_event_sec)

        events.append(
            AudioEvent(
                start=start_time,
                end=end_time,
                peak_time=peak_time,
                peak_score=float(scores[idx]),
            )
        )

        if max_events is not None and len(events) >= max_events:
            break

    # Sort chronologically for display
    events.sort(key=lambda e: e.start)
    return events



def detect_audio_events_for_clip(
    video_path: Path,
    sample_rate: int = 16_000,
    k: float = 1.8,
    pre_event_sec: float = 1.5,
    post_event_sec: float = 2.5,
    min_event_gap_sec: float = 1.0,
    max_events: int = 10,
) -> Tuple[List[AudioEvent], Dict[str, np.ndarray]]:
    """
    Given a video file, extract audio, compute features,
    and return audio events + diagnostics.
    """
    audio, sr = extract_mono_audio(video_path, sample_rate=sample_rate)
    clip_duration = float(audio.shape[0]) / float(sr)

    features = compute_audio_features(audio, sr)
    scores = compute_event_scores(features)
    events = detect_audio_events_from_scores(
        scores=scores,
        frame_times=features["frame_times"],
        clip_duration=clip_duration,
        k=k,
        pre_event_sec=pre_event_sec,
        post_event_sec=post_event_sec,
        min_event_gap_sec=min_event_gap_sec,
        max_events=max_events,
    )
    features["score"] = scores
    features["clip_duration"] = clip_duration
    return events, features


# ==========
# CLI driver
# ==========

def _format_events(events: List[AudioEvent]) -> str:
    if not events:
        return "  No salient audio events found."
    lines = []
    for e in events:
        lines.append("  " + str(e))
    return "\n".join(lines)

