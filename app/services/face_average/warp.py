import cv2
import numpy as np


def get_triangles(rect: tuple[int, int, int, int], points: np.ndarray) -> list[tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangle_list = subdiv.getTriangleList()
    triangles: list[tuple[int, int, int]] = []

    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

        if all(rect[0] <= p[0] <= rect[0] + rect[2] and rect[1] <= p[1] <= rect[1] + rect[3] for p in pts):
            idx: list[int] = []
            for p in pts:
                distances = np.sum((points.astype(np.float32) - np.float32(p)) ** 2, axis=1)
                nearest_idx = int(np.argmin(distances))
                if distances[nearest_idx] <= 9.0:
                    idx.append(nearest_idx)

            if len(idx) == 3 and len(set(idx)) == 3:
                triangles.append((idx[0], idx[1], idx[2]))

    return triangles


def affine_transform(src: np.ndarray, src_tri: list[tuple[float, float]], dst_tri: list[tuple[float, float]], size: tuple[int, int]) -> np.ndarray:
    mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(
        src,
        mat,
        (size[0], size[1]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_triangle(
    src_image: np.ndarray,
    dst_image: np.ndarray,
    src_triangle: list[tuple[int, int]],
    dst_triangle: list[tuple[int, int]],
) -> None:
    r1 = cv2.boundingRect(np.float32([src_triangle]))
    r2 = cv2.boundingRect(np.float32([dst_triangle]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = [(src_triangle[i][0] - r1[0], src_triangle[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(dst_triangle[i][0] - r2[0], dst_triangle[i][1] - r2[1]) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16)

    src_rect = src_image[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    warped_rect = affine_transform(src_rect, t1_rect, t2_rect, (r2[2], r2[3]))
    warped_rect = warped_rect * mask

    dst_slice = dst_image[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
    dst_image[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = dst_slice * (1 - mask) + warped_rect
