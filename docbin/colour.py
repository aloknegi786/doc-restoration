from __future__ import annotations

import cv2
import numpy as np

VALID_CHANNELS = ("gray", "red", "lab_l", "pca")


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_red(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, 2].copy()


def to_lab_l(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0]


def to_pca(bgr: np.ndarray) -> np.ndarray:
    pixels = bgr.reshape(-1, 3).astype(np.float64)
    mean = pixels.mean(axis=0)
    centered = pixels - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    pc1 = vecs[:, -1]
    proj = centered @ pc1

    p_min, p_max = proj.min(), proj.max()
    span = p_max - p_min
    if span > 0:
        proj = (proj - p_min) / span * 255.0
    else:
        proj[:] = 127.0
    result = proj.reshape(bgr.shape[:2]).astype(np.uint8)

    if np.median(result) < 128:
        result = 255 - result
    return result


def select_channel(bgr: np.ndarray, name: str) -> np.ndarray:
    if name == "gray":
        return to_gray(bgr)
    if name == "red":
        return to_red(bgr)
    if name == "lab_l":
        return to_lab_l(bgr)
    if name == "pca":
        return to_pca(bgr)
    raise ValueError(f"Unknown channel '{name}'. Valid: {VALID_CHANNELS}")
