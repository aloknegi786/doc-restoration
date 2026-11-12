from __future__ import annotations

import cv2
import numpy as np


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return binary.astype(np.uint8)
    best = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    return (labels == best).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    img_255 = (mask > 0).astype(np.uint8) * 255
    h, w = img_255.shape
    inv = cv2.bitwise_not(img_255)
    flooded = inv.copy()
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flooded[y, x] == 255:
            seed = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flooded, seed, (x, y), 128)
    holes = flooded == 255
    return ((img_255 > 0) | holes).astype(np.uint8)


def _coverage(mask: np.ndarray) -> float:
    return float(mask.sum()) / mask.size


def _pick_document_class(
    bright: np.ndarray, dark: np.ndarray, min_cov: float
) -> np.ndarray:
    b_lcc = _largest_cc(bright)
    d_lcc = _largest_cc(dark)
    b_cov = _coverage(b_lcc)
    d_cov = _coverage(d_lcc)

    if b_cov > 0.90 and d_cov >= min_cov:
        return d_lcc
    if d_cov > 0.90 and b_cov >= min_cov:
        return b_lcc
    return b_lcc if b_cov >= d_cov else d_lcc


def detect_roi(
    bgr: np.ndarray,
    *,
    margin_frac: float = 0.015,
    fallback_coverage: float = 0.95,
    min_coverage: float = 0.05,
    downsample: int = 4,
    close_frac: float = 0.04,
) -> np.ndarray:
    H, W = bgr.shape[:2]

    h = max(1, H // downsample)
    w = max(1, W // downsample)
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    sigma = max(1, min(h, w) // 20)
    ksize = int(sigma * 6) | 1
    L_blur = cv2.GaussianBlur(L, (ksize, ksize), sigma)

    _, thresh = cv2.threshold(
        L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bright = (thresh == 255).astype(np.uint8)
    dark = (1 - bright).astype(np.uint8)

    best = _pick_document_class(bright, dark, min_coverage)

    close_px = max(3, int(close_frac * h))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    closed = cv2.morphologyEx(best, cv2.MORPH_CLOSE, close_k)

    lcc = _largest_cc(closed)
    lcc = _fill_holes(lcc)

    margin_px = max(1, int(margin_frac * h))
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1)
    )
    eroded = cv2.erode(lcc, erode_k)

    roi = cv2.resize(eroded, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

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
    vis = bgr.copy()
    overlay = bgr.copy()

    overlay[roi_mask] = fill_color
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

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
    vis = bgr.copy()
    overlay = bgr.copy()
    overlay[mask] = fill_color
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis
