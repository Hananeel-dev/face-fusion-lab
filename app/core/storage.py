from app.core.config import COMPOSITE_DIR, FACE_DIR, TEMP_DIR


def ensure_storage_dirs() -> None:
    for directory in (FACE_DIR, COMPOSITE_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
