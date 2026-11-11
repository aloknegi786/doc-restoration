"""ROI detection: find the document region (page / palm leaf) within the frame.

Design decisions (binding for this project):
  - Channel: LAB L* (lightness).  For most datasets the document is lighter or
    darker than its surround; L* gives a one-channel signal that generalises
    across DIBCO (white page on dark border) and palm (tan leaf on grey surround).
  - Threshold: global Otsu on a heavily blurred L* channel.  Two large bimodal
    regions → Otsu is the right tool here.
  - Post-process: largest CC → fill interior holes → erode by a fixed fraction
    of image height (removes the dark leaf / page border).
  - Fallback: if the detected mask covers >95 % of the frame (nothing useful
    was separated), return a full-frame mask so the pipeline is a no-op.
  - Metrics are always scored over the whole image, never only inside the ROI.
    The ROI affects only which pixels are background-normalised / thresholded.

Public API:
    detect_roi(bgr, **kwargs) -> np.ndarray[bool]  (H, W), True = document
    draw_roi_overlay(bgr, roi_mask) -> np.ndarray  (H, W, 3) uint8
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _largest_cc(binary: np.ndarray) -> np.ndarray:
    """Return a uint8 mask containing only the largest connected component.

    Label 0 is OpenCV's background label and is always skipped.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return binary.astype(np.uint8)
    # stats rows: [left, top, width, height, area].  Skip label 0 (background).
    best = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    return (labels == best).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary uint8 mask (values 0 / 1).

    Strategy: invert the mask, flood-fill from each corner (external
    background), then every remaining non-zero pixel in the inverted image
    is an interior hole.
    """
    img_255 = (mask > 0).astype(np.uint8) * 255
    h, w = img_255.shape
    # Invert: foreground→0, background→255
    inv = cv2.bitwise_not(img_255)
    flooded = inv.copy()
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flooded[y, x] == 255:
            seed = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flooded, seed, (x, y), 128)
    # Pixels still 255 were unreachable from any border → holes
    holes = flooded == 255
    return ((img_255 > 0) | holes).astype(np.uint8)


def _coverage(mask: np.ndarray) -> float:
    return float(mask.sum()) / mask.size


def _pick_document_class(
    bright: np.ndarray, dark: np.ndarray, min_cov: float
) -> np.ndarray:
    """Choose whether bright or dark pixels represent the document.

    Heuristic (applied to the downsampled resolution):
      - Prefer the bright class (white/light page is the common case).
      - Switch to dark if the bright class's largest CC covers > 90 % of the
        frame (Otsu found nothing useful in that direction) and the dark class
        has a plausible coverage.
      - If both are extreme, return whichever has the larger CC.
    """
    b_lcc = _largest_cc(bright)
    d_lcc = _largest_cc(dark)
    b_cov = _coverage(b_lcc)
    d_cov = _coverage(d_lcc)

    if b_cov > 0.90 and d_cov >= min_cov:
        return d_lcc
    if d_cov > 0.90 and b_cov >= min_cov:
        return b_lcc
    return b_lcc if b_cov >= d_cov else d_lcc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_roi(
    bgr: np.ndarray,
    *,
    margin_frac: float = 0.015,
    fallback_coverage: float = 0.95,
    min_coverage: float = 0.05,
    downsample: int = 4,
    close_frac: float = 0.04,
) -> np.ndarray:
    """Detect the document / leaf region in a colour image.

    Parameters
    ----------
    bgr              : uint8 BGR image (H, W, 3).
    margin_frac      : erosion margin as a fraction of image height.
                       Trims dark page / leaf borders.
    fallback_coverage: if the final mask covers more than this fraction of
                       the frame the detection is considered a no-op and a
                       full-frame mask is returned.
    min_coverage     : if the mask is smaller than this fraction, detection
                       is also considered to have failed → full-frame mask.
    downsample       : factor for working at reduced resolution (speed +
                       noise suppression).  Final mask is upsampled back.
    close_frac       : morphological closing kernel size as a fraction of
                       the downsampled image height.

    Returns
    -------
    Boolean mask (H, W) — True inside the detected document region.
    """
    H, W = bgr.shape[:2]

    # 1. Downsample --------------------------------------------------------
    h = max(1, H // downsample)
    w = max(1, W // downsample)
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

    # 2. LAB L* channel ----------------------------------------------------
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]   # uint8 [0, 255] in OpenCV

    # 3. Gaussian blur (suppress texture, strokes, stains) -----------------
    sigma = max(1, min(h, w) // 20)
    ksize = (int(sigma * 6) | 1)   # nearest odd integer
    L_blur = cv2.GaussianBlur(L, (ksize, ksize), sigma)

    # 4. Otsu split on the blurred L channel --------------------------------
    _, thresh = cv2.threshold(
        L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bright = (thresh == 255).astype(np.uint8)
    dark = (1 - bright).astype(np.uint8)

    # 5. Choose which class is the document ---------------------------------
    best = _pick_document_class(bright, dark, min_coverage)

    # 6. Morphological closing to bridge small gaps -------------------------
    close_px = max(3, int(close_frac * h))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    closed = cv2.morphologyEx(best, cv2.MORPH_CLOSE, close_k)

    # 7. Largest CC + fill holes -------------------------------------------
    lcc = _largest_cc(closed)
    lcc = _fill_holes(lcc)

    # 8. Erode to strip dark border ----------------------------------------
    margin_px = max(1, int(margin_frac * h))
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
    )
    eroded = cv2.erode(lcc, erode_k)

    # 9. Upsample to original resolution ------------------------------------
    roi = cv2.resize(eroded, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # 10. Fallback if detection found nothing useful ------------------------
    cov = float(roi.sum()) / roi.size
    if cov > fallback_coverage or cov < min_coverage:
        return np.ones((H, W), dtype=bool)

    return roi


def draw_roi_overlay(
    bgr: np.ndarray,
    roi_mask: np.ndarray,
    fill_color: tuple[int, int, int] = (0, 200, 0),
    contour_color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.25,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Return a copy of `bgr` with the ROI mask overlaid.

    A semi-transparent filled region in `fill_color` is blended over the
    document area, and the boundary is drawn in `contour_color`.
    """
    vis = bgr.copy()
    overlay = bgr.copy()

    # Semi-transparent fill
    overlay[roi_mask] = fill_color
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Contour
    contours, _ = cv2.findContours(
        roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, contour_color, contour_thickness)

    return vis


def draw_mask_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    fill_color: tuple[int, int, int] = (0, 0, 200),
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a boolean mask on `bgr` with a flat semi-transparent colour."""
    vis = bgr.copy()
    overlay = bgr.copy()
    overlay[mask] = fill_color
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis

