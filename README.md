# Dashcam Video Pipeline

A CPU-friendly Python + FFmpeg pipeline for turning raw dashcam footage into
short, vertical, social-ready driving videos.

Designed to be:
- Fast
- Modular
- Debuggable
- Extensible

---

## Features
- Road + cabin clip pairing
- Motion-based event detection
- Audio-based salience detection
- Canonical event format
- JSON export for events
- Vertical video compositor with caption space
- FFmpeg progress bar
- CPU-only friendly

---

## Requirements
- Python 3.10+
- FFmpeg
- OpenCV
- NumPy
- tqdm (optional, for progress bars)

---

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Basic Usage
```bash
BASE="/media/matt/2004-1014/DCIM"

python -m src.run --base-dir "$BASE" ingest
python -m src.run --base-dir "$BASE" motion --index 7
python -m src.run --base-dir "$BASE" audio --index 7
python -m src.run --base-dir "$BASE" layout --index 7 --out output/test.mp4
```

---

## Output Artifacts
- Vertical MP4 videos
- Canonical event JSON files
- Optional NumPy debug arrays

---

## Design Philosophy
- Detect on original clips
- Canonicalize early
- Render fast
- Keep timelines explicit
- Avoid GPU dependencies

---

## Roadmap
- Event fusion (motion + audio)
- Event ranking
- Caption overlays
- Music alignment
- Batch / watcher mode
