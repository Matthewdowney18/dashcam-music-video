"""
layout_progress.py

Utilities for tracking and displaying progress during video layout and rendering
operations in the dashcam processing pipeline.

This module is responsible for:
- Parsing progress information from long-running FFmpeg processes
- Converting raw timing/frame data into human-readable progress metrics
- Rendering a lightweight progress indicator suitable for CLI use
- Remaining decoupled from video layout logic and FFmpeg invocation itself

Design goals:
- Minimal overhead (CPU-only environment, no GPU assumptions)
- Works with streamed FFmpeg stderr/stdout without requiring re-encoding
- Safe to use for long batch jobs (multiple clips, stacked layouts, captions)
- Easy to disable or swap out for other progress UIs in the future

Intended usage:
- Called by layout / render orchestration code
- Receives incremental FFmpeg output or timestamps
- Emits progress updates (percentage, ETA, elapsed time, etc.)

This module does NOT:
- Invoke FFmpeg directly
- Perform any video decoding or encoding
- Depend on GUI frameworks or external progress libraries

Keeping progress reporting isolated here allows layout, captioning,
and clip-selection logic to remain simple and testable.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import Optional


def _ffprobe_duration_seconds(path: Path) -> Optional[float]:
    """
    Return duration in seconds via ffprobe, or None if unknown.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        s = p.stdout.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _run_ffmpeg_with_progress(cmd: list[str], expected_duration_s: Optional[float]) -> None:
    """
    Run ffmpeg and show a progress bar based on out_time_ms if possible.
    """
    # Try tqdm first (nice progress bar), fallback to simple prints if unavailable.
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    # ffmpeg progress comes on stdout in key=value lines
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # keep errors for debugging
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if proc.stdout is None:
        raise RuntimeError("Failed to capture ffmpeg stdout for progress.")
    if proc.stderr is None:
        raise RuntimeError("Failed to capture ffmpeg stderr.")

    bar = None
    if tqdm is not None and expected_duration_s and expected_duration_s > 0:
        bar = tqdm(total=expected_duration_s, unit="s", dynamic_ncols=True)

    last_t = 0.0
    out_time_ms_re = re.compile(r"^out_time_ms=(\d+)$")

    try:
        for line in proc.stdout:
            line = line.strip()

            # Common keys: frame=, fps=, out_time_ms=, speed=, progress=
            m = out_time_ms_re.match(line)
            if m:
                out_ms = int(m.group(1))
                t = out_ms / 1_000_000.0  # microseconds → seconds
                if bar is not None:
                    delta = max(0.0, t - last_t)
                    if delta:
                        bar.update(delta)
                else:
                    # Fallback: print percent-ish if we can
                    if expected_duration_s and expected_duration_s > 0:
                        pct = min(100.0, (t / expected_duration_s) * 100.0)
                        sys.stdout.write(f"\r[ffmpeg] {pct:6.2f}%  ({t:6.1f}s / {expected_duration_s:6.1f}s)")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\r[ffmpeg] {t:6.1f}s")
                        sys.stdout.flush()
                last_t = t

            if line == "progress=end":
                break
    finally:
        if bar is not None:
            bar.close()

    stdout_rest = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()

    if bar is None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with code {rc}\n{stderr}")