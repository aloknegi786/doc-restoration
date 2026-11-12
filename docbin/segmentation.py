from __future__ import annotations

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    if gray.dtype != np.uint8:
        raise ValueError("otsu_threshold expects uint8")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0
    best = np.flatnonzero(sigma_b2 == sigma_b2.max())
    return int(round(float(best.mean())))


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    return gray <= otsu_threshold(gray)


def make_odd(w: int) -> int:
    w = int(round(w))
    return w if w % 2 == 1 else w + 1


def window_from_relative(shape: tuple[int, int], rel: float, floor: int = 15) -> int:
    return make_odd(max(floor, rel * min(shape)))


def local_mean_std(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
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


def sauvola_threshold(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    m, s = local_mean_std(gray, window)
    return m * (1.0 + k * (s / R - 1.0))


def sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    return gray <= sauvola_threshold(gray, window, k, R)
