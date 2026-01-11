# 📄 `PROJECT_CONTEXT.md`

```md
# PROJECT CONTEXT — Dashcam Video Pipeline

## Goal

Build a mostly automated system that converts raw dashcam footage into
short, social-ready driving videos (vertical, captionable, fast to render).

The pipeline should:
- Ingest raw dashcam clips
- Detect “interesting” moments (motion + audio)
- Canonicalize events into a shared timeline format
- Render vertical videos efficiently using FFmpeg
- Enable future steps: event fusion, caption overlays, music, ranking

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

---

## Architecture Overview

### Ingest
Pairs road + cabin clips by numeric ID.

**Module:** `src/ingest.py`

---

### Detection

Each detector operates on a *single video timeline* and outputs events
in seconds.

- **Motion detection**
  - Frame differencing
  - Detects motion spikes
  - Outputs `MotionEvent`

- **Audio detection**
  - FFmpeg audio extraction
  - RMS, spectral flux, band ratios
  - Salience scoring
  - Outputs `AudioEvent`

**Modules:**
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
Grouped as:

python
Copy code
EventSet(video_path, events)
Modules:

src/types/events.py

src/types/adapters.py

This enables:

merging motion + audio

ranking

overlays

JSON export

Rendering
Pure FFmpeg vertical compositor:

road on top

cabin on bottom

optional caption band

fast CPU encoding

progress bar

Modules:

src/layout.py

src/layout_progress.py

CLI
Unified entry point:


python -m src.run ...
Supports:

ingest

motion

audio

layout

canonical output

JSON export

Current Status
✅ Ingest stable
✅ Motion detection stable
✅ Audio detection stable
✅ Canonical events implemented
✅ JSON output implemented
✅ Vertical rendering optimized (pure FFmpeg)

Next milestones:

Combine motion + audio EventSets

Rank / fuse overlapping events

Caption overlays using canonical events


