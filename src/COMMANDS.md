# Dashcam Pipeline — Commands

```bash
source .venv/bin/activate
BASE="/media/matt/2004-1014/DCIM"

# ingest
python -m src.run --base-dir "$BASE" ingest --show 10

# motion
python -m src.run --base-dir "$BASE" motion --index 7 --camera both

# audio
python -m src.run --base-dir "$BASE" audio --index 7 --camera road

# layout render
python -m src.run --base-dir "$BASE" layout \
  --index 7 \
  --out output/test_vertical_7.mp4 \
  --seconds 10

# help
python -m src.run --help