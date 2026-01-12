# Dashcam Pipeline — Commands

Activate the virtual environment first:

```bash
source .venv/bin/activate
// Set the base directory where the dashcam SD card is mounted:


BASE="/media/matt/2004-1014/DCIM"


Ingest (pair road + cabin clips)
// Discover and pair clips by numeric ID (e.g. MOVA0009 ↔ MOVB0009):

python -m src.run --base-dir "$BASE" ingest --show 10
// Motion detection
//Detect motion spikes on a paired clip.

// Road + cabin:


python -m src.run --base-dir "$BASE" motion --index 7 --camera both
//Road only, canonical output:


python -m src.run --base-dir "$BASE" motion \
  --index 7 \
  --camera road \
  --canonical
//Road only, write canonical JSON:


python -m src.run --base-dir "$BASE" motion \
  --index 7 \
  --camera road \
  --canonical \
  --json output/events/motion_7_road.json
//Audio detection
//Detect salient audio events (honks, passing cars, speech, etc).

// Road only:

python -m src.run --base-dir "$BASE" audio --index 7 --camera road
Canonical + JSON output:


python -m src.run --base-dir "$BASE" audio \
  --index 7 \
  --camera road \
  --canonical \
  --json output/events/audio_7_road.json
//Debug feature arrays (saved as .npy):


python -m src.run --base-dir "$BASE" audio \
  --index 7 \
  --camera road \
  --debug-npy-prefix output/debug/audio_7
//Render vertical video (layout)
//Create a vertical composite (road top, cabin bottom).

Default preset (caption band included):

bash
Copy code
python -m src.run --base-dir "$BASE" layout \
  --index 7 \
  --out output/test_vertical_7.mp4 \
  --seconds 10
//Available presets:

legacy1080

caption1080

debug720

debug540

//Help

python -m src.run --help
yaml
Copy code

---