import cv2
import numpy as np


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = landmarks.astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    left_eye = points[33]
    right_eye = points[263]
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    scale = eye_distance if eye_distance > 1 else float(np.linalg.norm(centered, axis=1).mean())
    return centered / max(scale, 1.0)


def _score_from_distance(distance: float, good_distance: float) -> float:
    return round(float(np.clip(100.0 * (1.0 - distance / good_distance), 0.0, 100.0)), 1)


def compare_faces(
    landmarks_a: np.ndarray,
    landmarks_b: np.ndarray,
    mean_landmarks: np.ndarray,
    warped_a: np.ndarray,
    warped_b: np.ndarray,
) -> dict[str, float]:
    norm_a = _normalize_landmarks(landmarks_a)
    norm_b = _normalize_landmarks(landmarks_b)
    landmark_rms = float(np.sqrt(np.mean(np.sum((norm_a - norm_b) ** 2, axis=1))))
    landmark_similarity = _score_from_distance(landmark_rms, 0.35)

    warp_a = np.linalg.norm((landmarks_a - mean_landmarks).astype(np.float32), axis=1)
    warp_b = np.linalg.norm((landmarks_b - mean_landmarks).astype(np.float32), axis=1)
    warp_effort_px = float((warp_a.mean() + warp_b.mean()) / 2.0)
    warp_similarity = _score_from_distance(warp_effort_px / max(warped_a.shape[:2]), 0.055)

    gray_a = cv2.cvtColor(warped_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(gray_a - gray_b)))
    texture_similarity = _score_from_distance(mean_abs_diff, 90.0)

    pixel_std = float(np.std(warped_a.astype(np.float32) - warped_b.astype(np.float32)))
    composite_agreement = _score_from_distance(pixel_std, 85.0)

    overall = round(
        0.42 * landmark_similarity
        + 0.28 * warp_similarity
        + 0.20 * texture_similarity
        + 0.10 * composite_agreement,
        1,
    )

    return {
        "overall_similarity": overall,
        "landmark_geometry_similarity": landmark_similarity,
        "warp_similarity": warp_similarity,
        "texture_similarity": texture_similarity,
        "composite_agreement": composite_agreement,
        "landmark_rms": round(landmark_rms, 4),
        "average_warp_effort_px": round(warp_effort_px, 2),
        "mean_absolute_texture_difference": round(mean_abs_diff, 2),
        "pixel_variance": round(pixel_std, 2),
    }
