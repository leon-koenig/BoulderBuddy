from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

VIDEO_OUTPUT_DIR = OUTPUTS_DIR / "videos"
JSON_OUTPUT_DIR = OUTPUTS_DIR / "json"
DEBUG_OUTPUT_DIR = OUTPUTS_DIR / "debug"
