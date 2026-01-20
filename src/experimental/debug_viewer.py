"""
src/debug_viewer.py

Browser-based debug viewer:
- Produces browser-friendly road.mp4 + cabin.mp4 proxies in out_dir (remux if possible, else fast encode)
- Shows road (top) + cabin (bottom)
- Shows "conicals" lanes like layout.make_vertical_captioned_output_preset():
    MOTION ROAD, MOTION CABIN, AUDIO, SCORE
- Click event -> seek both videos
- Optional WebVTT overlay (playback overlay, not burned-in)
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from .ingest import DashcamConfig, discover_pairs
from .motion_events import detect_motion_events
from .audio_events import detect_audio_events_for_clip


# ---------------------------
# Conicals lane logic (copied from layout.py behavior)
# ---------------------------

@dataclass(frozen=True)
class OverlayEvent:
    start: float
    end: float
    peak: float
    label: str
    strength01: float = 1.0


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
            new_strength = max(cur.strength01, e.strength01)
            cur = OverlayEvent(new_start, new_end, new_peak, cur.label, new_strength)
        else:
            merged.append(cur)
            cur = e
    merged.append(cur)
    return merged


def _normalize_strength(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _build_union_windows(*lanes: List[OverlayEvent]) -> List[Tuple[float, float]]:
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
    best = 0.0
    for e in lane:
        if e.end >= start and e.start <= end:
            best = max(best, e.strength01)
    return best


def _best_peak_time_in_window(lane: List[OverlayEvent], start: float, end: float) -> Optional[float]:
    best_strength = -1.0
    best_peak = None
    for e in lane:
        if e.end >= start and e.start <= end:
            if e.strength01 > best_strength:
                best_strength = e.strength01
                best_peak = e.peak
    return best_peak


def _compute_score_lane(
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

    return _merge_overlapping(score_events)


# ---------------------------
# Browser proxy creation
# ---------------------------

def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _make_browser_mp4(src: Path, dst_mp4: Path) -> None:
    """
    Make a browser-playable mp4.
    1) Try remux (no reencode): ffmpeg -c copy
    2) If remux fails, fast encode (ultrafast x264 + aac)
    """
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Try stream copy (fastest, no reencode)
    try:
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?",   # video + optional audio
            "-c", "copy",
            "-movflags", "+faststart",
            str(dst_mp4),
        ])
        return
    except Exception:
        pass

    # Fallback: fast proxy encode
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst_mp4),
    ])


# ---------------------------
# VTT + HTML
# ---------------------------

def _sec_to_vtt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = ts % 60.0
    return f"{h:02}:{m:02}:{s:06.3f}"


def _write_vtt(events: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for i, e in enumerate(events, start=1):
        start = _sec_to_vtt(float(e["start"]))
        end = _sec_to_vtt(float(e["end"]))
        text = str(e.get("text", "")).strip() or "(event)"
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


HTML_TEMPLATE = Template(r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Dashcam Debug Viewer</title>
<style>
  body { margin:0; font-family: system-ui, sans-serif; }
  .toolbar {
    padding:8px; background:#222; color:#fff;
    display:flex; gap:8px; align-items:center;
  }
  .wrap { display:flex; height:calc(100vh - 42px); }
  .left { flex:1; display:flex; flex-direction:column; background:#111; }
  video { width:100%; height:50%; object-fit:contain; background:#000; }
  .right { width:420px; border-left:1px solid #ddd; overflow:auto; padding:12px; }
  .cap { padding:8px; margin:6px 0; border-radius:8px; cursor:pointer; }
  .cap:hover { background:#f2f2f2; }
  .cap.active { background:#dbeafe; }
  .meta { font-size:12px; opacity:0.7; }
  .tag { font-size:12px; padding:2px 6px; border-radius:999px; background:#eee; display:inline-block; margin-right:6px; }
</style>
</head>
<body>

<div class="toolbar">
  <button id="btnPlay">Play / Pause</button>
  <button id="btnBack">-1s</button>
  <button id="btnFwd">+1s</button>
  <span id="t">0.00</span>
</div>

<div class="wrap">
  <div class="left">
    <video id="road" controls muted playsinline>
      $road_track
    </video>
    <video id="cabin" controls muted playsinline></video>
  </div>
  <div class="right" id="caps"></div>
</div>

<script>
const road  = document.getElementById("road");
const cabin = document.getElementById("cabin");
const capsEl = document.getElementById("caps");
const tEl = document.getElementById("t");

road.src = "road.mp4";
cabin.src = "cabin.mp4";

function sync(master, slave) {
  const dt = Math.abs(master.currentTime - slave.currentTime);
  if (dt > 0.08) slave.currentTime = master.currentTime;
  if (!master.paused && slave.paused) slave.play().catch(()=>{});
  if (master.paused && !slave.paused) slave.pause();
}

document.getElementById("btnPlay").onclick = () => {
  if (road.paused) { road.play(); cabin.play().catch(()=>{}); }
  else { road.pause(); cabin.pause(); }
};
document.getElementById("btnBack").onclick = () => {
  road.currentTime = Math.max(0, road.currentTime - 1);
  cabin.currentTime = road.currentTime;
};
document.getElementById("btnFwd").onclick = () => {
  road.currentTime = road.currentTime + 1;
  cabin.currentTime = road.currentTime;
};

let events = [];
let nodes = [];

function highlight(t) {
  let idx = -1;
  for (let i=0; i<events.length; i++) {
    if (t >= events[i].start && t <= events[i].end) { idx = i; break; }
  }
  nodes.forEach((n,i)=>n.classList.toggle("active", i===idx));
  if (idx >= 0) {
    const n = nodes[idx];
    const r = n.getBoundingClientRect();
    const pr = capsEl.getBoundingClientRect();
    if (r.top < pr.top || r.bottom > pr.bottom) n.scrollIntoView({block:"center"});
  }
}

road.addEventListener("timeupdate", () => {
  tEl.textContent = road.currentTime.toFixed(2);
  sync(road, cabin);
  highlight(road.currentTime);
});
cabin.addEventListener("timeupdate", () => sync(cabin, road));

fetch("events.json")
  .then(r => r.json())
  .then(data => {
    events = data;
    nodes = events.map(e => {
      const d = document.createElement("div");
      d.className = "cap";
      const tag = e.kind ? `<span class="tag">${e.kind}</span>` : "";
      const score = (e.score !== null && e.score !== undefined) ? ` score=${Number(e.score).toFixed(3)}` : "";
      d.innerHTML = `
        <div class="meta">${tag}${Number(e.start).toFixed(2)} → ${Number(e.end).toFixed(2)}${score}</div>
        <div>${e.text || ""}</div>
      `;
      d.onclick = () => {
        road.currentTime = e.start;
        cabin.currentTime = e.start;
        road.play();
        cabin.play().catch(()=>{});
      };
      capsEl.appendChild(d);
      return d;
    });
  })
  .catch(err => {
    capsEl.textContent = "Failed to load events.json: " + err;
  });
</script>

</body>
</html>
""")


# ---------------------------
# Public API
# ---------------------------

def build_debug_viewer(
    base_dir: Path,
    pair_index: int = 0,
    out_dir: str | Path = "output/debug_viewer",
    max_duration: Optional[float] = 60.0,
    generate_vtt: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DashcamConfig(base_dir=Path(base_dir))
    pairs = discover_pairs(cfg)
    if not pairs:
        raise RuntimeError("No clip pairs found. base_dir must contain DCIMA/ and DCIMB/.")

    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index {pair_index} out of range (found {len(pairs)})")

    pair = pairs[pair_index]
    road_src = Path(pair.road)
    cabin_src = Path(pair.cabin)

    # Browser-friendly proxies
    road_mp4 = out_dir / "road.mp4"
    cabin_mp4 = out_dir / "cabin.mp4"
    _make_browser_mp4(road_src, road_mp4)
    _make_browser_mp4(cabin_src, cabin_mp4)

    # --- detect events exactly like make_vertical_captioned_output_preset() ---

    road_motion = detect_motion_events(
        video_path=road_src,
        downscale_width=320,
        frame_step=2,
        pre_event_sec=3.0,
        post_event_sec=5.0,
        min_event_gap_sec=5.0,
        threshold_std=2.0,
        max_events=20,
    )
    cabin_motion = detect_motion_events(
        video_path=cabin_src,
        downscale_width=320,
        frame_step=2,
        pre_event_sec=3.0,
        post_event_sec=5.0,
        min_event_gap_sec=5.0,
        threshold_std=2.0,
        max_events=20,
    )

    audio_events, _features = detect_audio_events_for_clip(
        video_path=road_src,  # audio assumed same
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
    score_lane = _compute_score_lane(road_lane, cabin_lane, audio_lane)

    # trim to max_duration (same idea as layout.py)
    if max_duration is not None:
        md = float(max_duration)

        def clip_lane(lane: List[OverlayEvent]) -> List[OverlayEvent]:
            out: List[OverlayEvent] = []
            for e in lane:
                if e.start >= md:
                    continue
                out.append(OverlayEvent(
                    start=max(0.0, e.start),
                    end=min(md, e.end),
                    peak=min(md, max(0.0, e.peak)),
                    label=e.label,
                    strength01=e.strength01,
                ))
            return out

        road_lane = clip_lane(road_lane)
        cabin_lane = clip_lane(cabin_lane)
        audio_lane = clip_lane(audio_lane)
        score_lane = clip_lane(score_lane)

    # Flatten events for UI list (but keep lane labels exactly)
    def lane_to_items(kind: str, lane: List[OverlayEvent]) -> List[Dict[str, Any]]:
        return [{
            "kind": kind,
            "start": e.start,
            "end": e.end,
            "score": e.strength01,
            "text": e.label,
        } for e in lane]

    items: List[Dict[str, Any]] = []
    items += lane_to_items("MOTION ROAD", road_lane)
    items += lane_to_items("MOTION CABIN", cabin_lane)
    items += lane_to_items("AUDIO", audio_lane)
    items += lane_to_items("SCORE", score_lane)
    items.sort(key=lambda x: (x["start"], x["end"]))

    (out_dir / "events.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    # Optional VTT overlay (shows the same labels)
    road_track = ""
    if generate_vtt:
        _write_vtt(items, out_dir / "events.vtt")
        road_track = '<track kind="subtitles" src="events.vtt" srclang="en" default>'

    html = HTML_TEMPLATE.safe_substitute(road_track=road_track)
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")

    return out_dir


def print_how_to_open(out_dir: Path) -> None:
    print("\n[debug-viewer] ready")
    print(f"  out_dir: {out_dir}")
    print("  open via a local server:")
    print(f"    cd {out_dir}")
    print("    python3 -m http.server 8001")
    print("    open: http://localhost:8001/viewer.html\n")
