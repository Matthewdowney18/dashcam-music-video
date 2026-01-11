# src/types/events.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# Canonical enums / tags used across modules
EventType = Literal["motion", "audio", "combined", "custom"]
Camera = Literal["road", "cabin", "both", "unknown"]


@dataclass(frozen=True)
class Event:
    """
    Canonical representation of a detected "interesting moment" on a timeline.

    Times are in seconds relative to the *video timeline this event was detected on*.
    For overlays on vertical_test_output.mp4, you want events detected on that same file,
    or you need a mapping step.
    """
    # Timeline
    start_s: float
    end_s: float

    # Identity
    event_type: EventType
    label: str  # short display label, e.g. "MOTION ↑", "AUDIO ↑"
    camera: Camera = "unknown"

    # Ranking / scoring
    score: float = 0.0                  # recommend normalized to [0..1], not required
    confidence: Optional[float] = None  # optional normalized [0..1]

    # Provenance / debug
    source: str = ""                    # e.g. "motion.simple_v1"
    meta: Dict[str, Any] = field(default_factory=dict)

    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def validate(self) -> None:
        if self.start_s < 0 or self.end_s < 0:
            raise ValueError(
                f"Event times must be >= 0 (got start_s={self.start_s}, end_s={self.end_s})"
            )
        if self.end_s < self.start_s:
            raise ValueError(
                f"Event end_s must be >= start_s (got start_s={self.start_s}, end_s={self.end_s})"
            )

    def __str__(self) -> str:
        s = (
            f"{self.start_s:6.2f}s → {self.end_s:6.2f}s"
            f"  [{self.event_type}]"
            f"  {self.label}"
            f"  cam={self.camera}"
        )
        if self.score is not None:
            s += f"  score={self.score:.3f}"
        if self.confidence is not None:
            s += f"  conf={self.confidence:.3f}"
        if self.source:
            s += f"  src={self.source}"
        return s


@dataclass(frozen=True)
class EventSet:
    """
    A set of events that correspond to a specific video timeline.
    """
    video_path: str
    events: List[Event]

    def validated(self) -> "EventSet":
        for e in self.events:
            e.validate()
        return self
