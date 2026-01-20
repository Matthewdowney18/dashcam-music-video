# PROJECT CONTEXT — Dashcam Video Pipeline

## Roles
### Matt = Director
- Decides intent
- Describes goals
- Chooses architecture direction

### ChatGPT = Architect / Research / System Design
- Pipeline design
- Module boundaries
- Algorithmic decisions
- Performance + UX tradeoffs
- Refactor planning
- Integration strategy

### Codex (in Cursor) = Hands
- Edits code in repo
- Scaffolds modules
- Applies diffs
- Writes boilerplate
- Maintains file consistency

---

## Goal
Build a mostly automated system that converts raw dashcam footage into
short, social-ready driving videos (vertical, captionable, fast to render).

The pipeline should:
- Ingest raw dashcam clips and GPS data
- Detect “interesting” moments (motion + audio)
- Canonicalize events into a shared timeline format
- Render vertical videos efficiently using FFmpeg
- Enable future steps: event fusion, caption overlays, music, ranking
- Be reasonably fast on a CPU-only laptop

---

## Hardware / Constraints
- OS: Linux (Ubuntu)
- Machine: Dell XPS 13 (CPU-only)
- RAM: 16GB
- No GPU
- Must be fast enough for batch processing
- FFmpeg available

---

## Dashcam Assumptions
- Two cameras:
  - **Road** (front): `DCIMA/`
  - **Cabin** (interior): `DCIMB/`
- Clips are ~1 minute long
- Filenames share a numeric suffix:
  - `MOVA0009.avi` ↔ `MOVB0009.avi`
- Audio is identical between cameras (currently assumed)
- Folders are located at `/media/matt/2004-1014/DCIM/`

---

## Architecture Overview
### Ingest
Pairs road + cabin clips by numeric ID.

Module:
- `src/ingest.py`

---

### Detection
Each detector operates on a single video timeline and outputs events in
seconds.

- **Motion detection**
  - Frame differencing
  - Detects motion spikes
  - Outputs `MotionEvent`
- **Audio detection**
  - FFmpeg audio extraction
  - RMS, spectral flux, band ratios
  - Salience scoring
  - Outputs `AudioEvent`

Modules:
- `src/motion_events.py`
- `src/audio_events.py`

---

### Canonical Events
All detections are converted into a shared format:

```python
Event(
  start_s,
  end_s,
  event_type,
  label,
  camera,
  score,
  confidence,
  source,
  meta
)
```

Grouped as:
- `EventSet(video_path, events)`

Modules:
- `src/types/events.py`
- `src/types/adapters.py`

This enables:
- Merging motion + audio
- Ranking
- Overlays
- JSON export

---

### Rendering
Pure FFmpeg vertical compositor:
- Road on top
- Cabin on bottom
- Optional caption band
- Fast CPU encoding
- Progress bar

Modules:
- `src/layout.py`
- `src/layout_progress.py`

---

### CLI
Unified entry point:
- `python -m src.run ...`

Supports:
- `ingest`
- `motion`
- `audio`
- `layout`
- Canonical output
- JSON export

---

## Current Status
- ✅ Ingest stable
- ✅ Motion detection stable
- ✅ Audio detection stable
- ✅ Canonical events implemented
- ✅ JSON output implemented
- ✅ Vertical rendering optimized (pure FFmpeg)

---

## Next Milestones
- Combine motion + audio `EventSet`s
- Rank / fuse overlapping events
- Caption overlays using canonical events
