import cv2
import numpy as np

from app.services.face_average.warp import get_triangles, warp_triangle


def _boundary_points(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            (0, 0),
            (width // 2, 0),
            (width - 1, 0),
            (width - 1, height // 2),
            (width - 1, height - 1),
            (width // 2, height - 1),
            (0, height - 1),
            (0, height // 2),
        ],
        dtype=np.int32,
    )


def warp_face_to_landmarks(image: np.ndarray, src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    src_points = np.vstack([src_landmarks, _boundary_points(w, h)]).astype(np.int32)
    dst_points = np.vstack([dst_landmarks, _boundary_points(w, h)]).astype(np.int32)
    warped = np.zeros((h, w, 3), dtype=np.float32)

    triangles = get_triangles((0, 0, w, h), dst_points)
    for i, j, k in triangles:
        src_triangle = [tuple(src_points[i]), tuple(src_points[j]), tuple(src_points[k])]
        dst_triangle = [tuple(dst_points[i]), tuple(dst_points[j]), tuple(dst_points[k])]
        warp_triangle(image, warped, src_triangle, dst_triangle)

    return np.clip(warped, 0, 255).astype(np.uint8)
