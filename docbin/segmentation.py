"""Binarization methods, implemented from scratch.

Convention used everywhere in this project:
    INK IS True (foreground). Paper is False.

All threshold functions take an 8-bit grayscale image (H, W) uint8.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Otsu (global)
# --------------------------------------------------------------------------
def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's method: pick t maximising between-class variance.

    Pixels with value <= t are class 0 (dark, i.e. ink); > t are class 1.
    """
    if gray.dtype != np.uint8:
        raise ValueError("otsu_threshold expects uint8")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)                       # class-0 probability
    mu = np.cumsum(p * np.arange(256))         # class-0 cumulative mean
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0     # empty classes contribute 0
    # Ties (plateau of equal maxima) are resolved by taking the middle.
    best = np.flatnonzero(sigma_b2 == sigma_b2.max())
    return int(round(float(best.mean())))


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Global Otsu. Returns ink mask (True = ink)."""
    return gray <= otsu_threshold(gray)


# --------------------------------------------------------------------------
# Local statistics via integral images (O(1) per pixel for any window size)
# --------------------------------------------------------------------------
def make_odd(w: int) -> int:
    w = int(round(w))
    return w if w % 2 == 1 else w + 1


def window_from_relative(shape: tuple[int, int], rel: float, floor: int = 15) -> int:
    """Placeholder window-size rule: a fraction of the shorter image side.

    This stands in for a proper stroke-width estimate (planned next). Kept
    in one place so every script (baseline, sweep, pipeline) uses the same
    rule and a later swap only touches this function.
    """
    return make_odd(max(floor, rel * min(shape)))


def local_mean_std(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and (population) std in a window x window neighbourhood.

    Uses two integral images (sum and sum of squares) in float64.
    Borders are handled by reflect padding.
    """
    window = make_odd(window)
    pad = window // 2
    g = np.pad(gray.astype(np.float64), pad, mode="reflect")
    h, w = g.shape

    s = np.zeros((h + 1, w + 1))
    s2 = np.zeros((h + 1, w + 1))
    s[1:, 1:] = g.cumsum(0).cumsum(1)
    s2[1:, 1:] = (g * g).cumsum(0).cumsum(1)

    def box(ii: np.ndarray) -> np.ndarray:
        return ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]

    n = float(window * window)
    mean = box(s) / n
    var = np.maximum(box(s2) / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


# --------------------------------------------------------------------------
# Local thresholding
# --------------------------------------------------------------------------
def niblack_threshold(gray: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
    """T(x,y) = m(x,y) + k * s(x,y). Ink where gray <= T.

    Original Niblack convention (k is negative, e.g. -0.2). Note that
    scikit-image uses T = m - k*s, so its k=+0.2 equals k=-0.2 here.
    """
    m, s = local_mean_std(gray, window)
    return m + k * s


def sauvola_threshold(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1)). Ink where gray <= T."""
    m, s = local_mean_std(gray, window)
    return m * (1.0 + k * (s / R - 1.0))


def niblack_binarize(gray: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
    return gray <= niblack_threshold(gray, window, k)


def sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    return gray <= sauvola_threshold(gray, window, k, R)
