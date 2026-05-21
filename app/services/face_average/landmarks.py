import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.core.config import LANDMARKER_MODEL_PATH

_base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL_PATH))
_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
_detector = vision.FaceLandmarker.create_from_options(_options)


def detect_landmarks(image: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w, _ = image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]
    return np.array(coords)
