import cv2
import numpy as np

from app.core.config import FACE_SIZE
from app.services.face_average.landmarks import detect_landmarks


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _clamp_bbox(x: int, y: int, w: int, h: int, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ih, iw = image_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return x, y, w, h


def _landmark_bbox(landmarks: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    x_min = int(np.min(xs))
    x_max = int(np.max(xs))
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    return _clamp_bbox(x_min, y_min, x_max - x_min, y_max - y_min, image_shape)


def _detect_face_bbox_haar(image: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        return None

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
    return _clamp_bbox(int(x), int(y), int(w), int(h), image.shape)


def _expand_to_square(x: int, y: int, w: int, h: int, image_shape: tuple[int, int, int], margin: float = 0.55) -> tuple[int, int, int, int]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = int(max(w, h) * (1.0 + margin))
    return _clamp_bbox(int(cx - side / 2.0), int(cy - side / 2.0), side, side, image_shape)


def _align_face_crop(face_crop: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )
    rot = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    return cv2.warpAffine(face_crop, rot, (face_crop.shape[1], face_crop.shape[0]), flags=cv2.INTER_CUBIC)


def extract_face(image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    landmarks = detect_landmarks(image)
    strategy = "mediapipe_landmark_crop"

    if landmarks is not None:
        bbox = _landmark_bbox(landmarks, image.shape)
    else:
        bbox = _detect_face_bbox_haar(image)
        strategy = "haar_fallback_crop"
        if bbox is None:
            return None, None, "face_not_detected"

    x, y, w, h = _expand_to_square(*bbox, image.shape)
    crop = image[y : y + h, x : x + w]
    crop_landmarks = detect_landmarks(crop)
    if crop_landmarks is None:
        resized = cv2.resize(crop, FACE_SIZE)
        resized_landmarks = detect_landmarks(resized)
        return resized, resized_landmarks, strategy

    aligned = _align_face_crop(crop, crop_landmarks)
    resized = cv2.resize(aligned, FACE_SIZE)
    resized_landmarks = detect_landmarks(resized)
    if resized_landmarks is None:
        return None, None, "landmarks_lost_after_alignment"

    return resized, resized_landmarks, strategy
