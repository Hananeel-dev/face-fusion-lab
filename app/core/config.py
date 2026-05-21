from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "storage"
FACE_DIR = DATA_DIR / "faces"
COMPOSITE_DIR = DATA_DIR / "composites"
TEMP_DIR = DATA_DIR / "temp"

FACE_SIZE = (256, 256)

# Keep path absolute so it works regardless of current working directory.
LANDMARKER_MODEL_PATH = BASE_DIR / "app" / "models" / "face_landmarker.task"

API_PREFIX = "/api/v1"
