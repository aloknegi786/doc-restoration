"""Channel selection: convert BGR to a single-channel uint8 for binarization.

Four options are compared empirically on the tuning set (see eval/ablation.py):
  gray   – standard luminance-weighted grayscale (cv2.cvtColor BGR→GRAY)
  red    – the R channel (index 2 in BGR).  Often the least contaminated
           by bleed-through, whose ink was typically blue/black.
  lab_l  – L* from CIELAB.  Perceptually linear; better than gray under
           coloured stains or non-neutral illumination.
  pca    – first principal component of the pixel distribution.  Finds
           the linear combination of R/G/B that explains the most
           ink-vs-paper variance; adaptive to each image.

All functions:
  - Accept uint8 BGR arrays (H, W, 3).
  - Return uint8 arrays (H, W) where LOWER values = more ink.
    (Consistent with every other module: ink is True after thresholding.)
"""
from __future__ import annotations

import cv2
import numpy as np

VALID_CHANNELS = ("gray", "red", "lab_l", "pca")


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Standard OpenCV BGR-to-gray (luminance-weighted)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_red(bgr: np.ndarray) -> np.ndarray:
    """Red channel.  OpenCV stores BGR, so index 2 is red."""
    return bgr[:, :, 2].copy()


def to_lab_l(bgr: np.ndarray) -> np.ndarray:
    """L* channel from CIELAB.  Returned as uint8 [0, 255]."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0]   # OpenCV scales L* to [0, 255]


def to_pca(bgr: np.ndarray) -> np.ndarray:
    """Project onto the first principal component of the BGR pixel distribution.

    The sign is chosen so that paper (the majority class) has HIGH values and
    ink has LOW values, matching the convention of every other channel function.
    The output is rescaled to [0, 255] uint8.
    """
    pixels = bgr.reshape(-1, 3).astype(np.float64)
    mean = pixels.mean(axis=0)
    centered = pixels - mean
    # Covariance matrix (3x3); use float64 for numerical stability
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    # eigh returns eigenvalues in ascending order; last column = first PC
    _, vecs = np.linalg.eigh(cov)
    pc1 = vecs[:, -1]           # (3,) first principal component
    proj = centered @ pc1       # (N,) signed projection

    # Rescale to [0, 255]
    p_min, p_max = proj.min(), proj.max()
    span = p_max - p_min
    if span > 0:
        proj = (proj - p_min) / span * 255.0
    else:
        proj[:] = 127.0
    result = proj.reshape(bgr.shape[:2]).astype(np.uint8)

    # Ensure paper (high-value majority) stays high.
    # If the median < 128, the PC points the wrong way → invert.
    if np.median(result) < 128:
        result = 255 - result
    return result


def select_channel(bgr: np.ndarray, name: str) -> np.ndarray:
    """Dispatch to the named channel function.

    Parameters
    ----------
    bgr  : uint8 BGR image (H, W, 3).
    name : one of 'gray', 'red', 'lab_l', 'pca'.

    Returns
    -------
    uint8 single-channel image (H, W).
    """
    if name == "gray":
        return to_gray(bgr)
    if name == "red":
        return to_red(bgr)
    if name == "lab_l":
        return to_lab_l(bgr)
    if name == "pca":
        return to_pca(bgr)
    raise ValueError(f"Unknown channel '{name}'. Valid: {VALID_CHANNELS}")

