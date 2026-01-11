from __future__ import annotations

from typing import List

from .events import Event, Camera
from ..motion_events import MotionEvent  # this imports from src/events.py (motion module)
from ..audio_events import AudioEvent


def motion_events_to_canonical(
    motion_events: List[MotionEvent],
    *,
    camera: Camera,
    source: str = "motion.simple_v1",
    label: str = "MOTION ↑",
) -> List[Event]:
    out: List[Event] = []
    for me in motion_events:
        out.append(
            Event(
                start_s=float(me.start_time),
                end_s=float(me.end_time),
                event_type="motion",
                label=label,
                camera=camera,
                score=float(me.score),
                confidence=None,
                source=source,
                meta={
                    "peak_time": float(me.peak_time),
                    "raw_score": float(me.score),
                },
            )
        )
    return out


def audio_events_to_canonical(
    audio_events: List[AudioEvent],
    *,
    camera: Camera,
    source: str = "audio.salience_v1",
    label: str = "AUDIO ↑",
) -> List[Event]:
    out: List[Event] = []
    for ae in audio_events:
        out.append(
            Event(
                start_s=float(ae.start),
                end_s=float(ae.end),
                event_type="audio",
                label=label,
                camera=camera,
                score=float(ae.peak_score),
                confidence=None,
                source=source,
                meta={
                    "peak_time": float(ae.peak_time),
                    "raw_score": float(ae.peak_score),
                },
            )
        )
    return out
