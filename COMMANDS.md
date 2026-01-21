# Dashcam Pipeline — Commands

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Set the base directory where the dashcam SD card is mounted:

```bash
BASE="/media/matt/2004-1014/DCIM"
```

---

## Ingest (pair road + cabin clips)
Discover and pair clips by numeric ID (e.g. `MOVA0009` ↔ `MOVB0009`).

```bash
python -m src.run --base-dir "$BASE" ingest --show 10
```

---

## Motion Detection
Detect motion spikes on a paired clip.

Road + cabin:
```bash
python -m src.run --base-dir "$BASE" motion --index 7 --camera both
```

Road only, canonical output:
```bash
python -m src.run --base-dir "$BASE" motion \
  --index 7 \
  --camera road \
  --canonical
```

Road only, write canonical JSON:
```bash
python -m src.run --base-dir "$BASE" motion \
  --index 7 \
  --camera road \
  --canonical \
  --json output/events/motion_7_road.json
```

---

## Audio Detection
Detect salient audio events (honks, passing cars, speech, etc).

Road only:
```bash
python -m src.run --base-dir "$BASE" audio --index 7 --camera road
```

Canonical + JSON output:
```bash
python -m src.run --base-dir "$BASE" audio \
  --index 7 \
  --camera road \
  --canonical \
  --json output/events/audio_7_road.json
```

Debug feature arrays (saved as `.npy`):
```bash
python -m src.run --base-dir "$BASE" audio \
  --index 7 \
  --camera road \
  --debug-npy-prefix output/debug/audio_7
```

---

## Render Vertical Video (Layout)
Create a vertical composite (road top, cabin bottom).

Default preset (caption band included):
```bash
python -m src.run --base-dir "$BASE" layout \
  --index 7 \
  --out output/test_vertical_7.mp4 \
  --seconds 10
```

Available presets:
- `legacy1080`
- `caption1080`
- `debug720`
- `debug540`

---

## Render Captioned (Vertical)
Caption band with event overlays.

```bash
python -m src.run --base-dir "$BASE" captioned \
  --index 7 \
  --out output/captioned_7.mp4 \
  --seconds 10 \
  --preset debug720 \
  --detect-camera road
```

---

## Render Captioned + Side Panel
Caption band plus a right-side panel for future annotations.
Defaults: `--quality debug` uses faster encode, smaller panel, and fewer labels.

```bash
python -m src.run --base-dir "$BASE" captioned-panel \
  --index 7 \
  --out output/captioned_panel_7.mp4 \
  --seconds 10 \
  --preset debug720 \
  --quality debug \
  --panel-width 420 \
  --stack-width 960 \
  --detect-camera road
```

Final quality (slower, full labels):
```bash
python -m src.run --base-dir "$BASE" captioned-panel \
  --index 7 \
  --out output/captioned_panel_7_final.mp4 \
  --seconds 10 \
  --preset debug720 \
  --quality final \
  --detect-camera road
```

---

## Project Export
Compress repo to upload to chatGPT.
```bash
make -f tools/project_export/Makefile export
```


## Help
```bash
python -m src.run --help
```
