"""Small-component removal (morphological cleanup).

After binarization some isolated noise specks appear as small connected
components in the ink mask.  Removing them reduces false positives without
erasing real ink, provided the size threshold is kept low enough to
preserve pencil marks and tiny dots that appear in the ground truth.

Binding rule (from project scope): use a *tuned, low* threshold and
keep any closing minimal; never use morphological opening, which erases
thin strokes and pencil marks.

Public API:
    remove_small_components(mask, min_size) -> np.ndarray[bool]
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 15,
) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels.

    Parameters
    ----------
    mask     : bool (or uint8) ink mask (H, W) — True / 255 = ink.
    min_size : minimum component area in pixels.  Components strictly
               smaller than this are removed.  Keep this low (default 15)
               to preserve thin pencil marks and small genuine ink dots.

    Returns
    -------
    Boolean mask (H, W) with small components removed.
    """
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # Label 0 is OpenCV's background — always keep background (False in mask).
    # Collect labels whose area >= min_size.
    keep = np.zeros(n, dtype=bool)
    keep[0] = False   # never keep background label as ink
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            keep[label] = True

    return keep[labels].astype(bool)

