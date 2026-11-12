from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 15,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    keep = np.zeros(n, dtype=bool)
    keep[0] = False
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            keep[label] = True

    return keep[labels].astype(bool)
