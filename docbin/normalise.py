from __future__ import annotations

import cv2
import numpy as np


def estimate_background(
    gray: np.ndarray,
    roi: np.ndarray,
    *,
    sigma_frac: float = 0.10,
    downsample: int = 4,
) -> np.ndarray:
    H, W = gray.shape
    radius_full = max(10.0, sigma_frac * min(H, W))

    work = gray.copy()
    work[~roi] = 255

    h_s = max(1, H // downsample)
    w_s = max(1, W // downsample)
    small = cv2.resize(work, (w_s, h_s), interpolation=cv2.INTER_AREA)

    radius_s = max(3, int(radius_full / downsample))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius_s * 2 + 1, radius_s * 2 + 1)
    )
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)

    bg = cv2.resize(closed.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 1.0).astype(np.float32)


def normalise_background(
    gray: np.ndarray,
    roi: np.ndarray,
    *,
    sigma_frac: float = 0.10,
    downsample: int = 4,
) -> np.ndarray:
    bg = estimate_background(gray, roi, sigma_frac=sigma_frac, downsample=downsample)
    normalised = np.clip(255.0 * gray.astype(np.float32) / bg, 0.0, 255.0)
    result = normalised.astype(np.uint8)
    result[~roi] = 255
    return result
