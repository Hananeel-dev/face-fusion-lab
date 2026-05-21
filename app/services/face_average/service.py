import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import COMPOSITE_DIR
from app.services.face_average.engine import warp_face_to_landmarks
from app.services.face_average.metrics import compare_faces
from app.services.face_average.preprocess import decode_image, extract_face


class FaceProcessingError(Exception):
    pass


def _read_face(file_bytes: bytes, label: str) -> tuple[np.ndarray, np.ndarray, str]:
    image = decode_image(file_bytes)
    if image is None:
        raise FaceProcessingError(f"{label} is not a valid image.")

    face, landmarks, strategy = extract_face(image)
    if face is None or landmarks is None:
        raise FaceProcessingError(f"No clear face was detected in {label}. Try a sharper, front-facing photo.")

    return face, landmarks, strategy


def create_two_face_composite(image_a_bytes: bytes, image_b_bytes: bytes) -> dict[str, object]:
    face_a, landmarks_a, strategy_a = _read_face(image_a_bytes, "image_a")
    face_b, landmarks_b, strategy_b = _read_face(image_b_bytes, "image_b")

    mean_landmarks = ((landmarks_a.astype(np.float32) + landmarks_b.astype(np.float32)) / 2.0).astype(np.float32)
    warped_a = warp_face_to_landmarks(face_a, landmarks_a, mean_landmarks)
    warped_b = warp_face_to_landmarks(face_b, landmarks_b, mean_landmarks)
    composite = np.clip((warped_a.astype(np.float32) + warped_b.astype(np.float32)) / 2.0, 0, 255).astype(np.uint8)

    filename = f"two_face_composite_{uuid.uuid4()}.png"
    output_path = COMPOSITE_DIR / filename
    cv2.imwrite(str(output_path), composite)

    metrics = compare_faces(landmarks_a, landmarks_b, mean_landmarks, warped_a, warped_b)
    return {
        "file": filename,
        "path": output_path,
        "metrics": metrics,
        "strategies": {
            "image_a": strategy_a,
            "image_b": strategy_b,
            "warp": "delaunay_landmark_warp_to_shared_mean_shape",
            "blend": "equal_weight_50_50_pixel_average",
        },
    }


def get_composite_path(filename: str) -> Path:
    return COMPOSITE_DIR / filename
