# Dashcam Music Video – Useful Commands

```bash

## setup venv
source .venv/bin/activate

## Ingest
python -m src.ingest --base-dir "/media/matt/2004-1014/DCIM"

## detect motion spikes in a clip
python -m src.events --base-dir "/media/matt/2004-1014/DCIM" --index 7 

## detect smooth flow segments 
python -m src.events --base-dir "/media/matt/2004-1014/DCIM" --index 7 --mode smooth

## combine two cameras into one picture
python -m src.layout \
    --base-dir "/media/matt/2004-1014/DCIM" \
    --out "output/test/test_vertical_7.mp4" \
    --index 7 \
    --seconds 60