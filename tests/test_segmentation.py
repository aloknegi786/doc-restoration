"""Unit tests for the segmentation module (docbin/segmentation.py)."""
from __future__ import annotations

import numpy as np
import pytest

from docbin.segmentation import (
    local_mean_std,
    make_odd,
    otsu_binarize,
    otsu_threshold,
    sauvola_binarize,
    sauvola_threshold,
    window_from_relative,
)


def _synthetic_doc(h: int = 200, w: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dark strokes on paper with an illumination gradient from left to right."""
    rng = np.random.default_rng(seed)
    bg = np.tile(np.linspace(120, 230, w), (h, 1))
    img = bg.copy()
    ink = np.zeros((h, w), dtype=bool)
    for y in range(30, h - 20, 40):
        ink[y : y + 3, 20 : w - 20] = True
        ink[y - 8 : y + 8, 60:63] = True
    img[ink] *= 0.35
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), ink


# --------------------------------------------------------------------------
# Otsu
# --------------------------------------------------------------------------

def test_otsu_bimodal_exact():
    g = np.concatenate([np.full(500, 50), np.full(500, 200)]).astype(np.uint8).reshape(20, 50)
    t = otsu_threshold(g)
    assert 50 <= t < 200


def test_otsu_non_uint8_raises():
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((10, 10), dtype=np.float32))


def test_otsu_binarize_produces_boolean_mask():
    g, _ = _synthetic_doc(50, 50)
    mask = otsu_binarize(g)
    assert mask.dtype == bool
    assert mask.shape == g.shape
    assert mask.any() and not mask.all()


# --------------------------------------------------------------------------
# Integral-image local statistics
# --------------------------------------------------------------------------

def test_make_odd():
    assert make_odd(10) == 11
    assert make_odd(11) == 11
    assert make_odd(10.2) == 11
    assert make_odd(10.8) == 11


def test_window_from_relative():
    shape = (500, 1000)
    # 0.05 * 500 = 25 -> 25 (odd)
    assert window_from_relative(shape, rel=0.05, floor=15) == 25
    # floor enforcement
    assert window_from_relative(shape, rel=0.001, floor=15) == 15


def test_local_mean_std_matches_bruteforce():
    g, _ = _synthetic_doc(60, 80)
    w = 11
    m, s = local_mean_std(g, w)
    pad = w // 2
    gp = np.pad(g.astype(float), pad, mode="reflect")
    for y, x in [(0, 0), (5, 70), (30, 40), (59, 79)]:
        win = gp[y : y + w, x : x + w]
        assert m[y, x] == pytest.approx(win.mean(), abs=1e-6)
        assert s[y, x] == pytest.approx(win.std(), abs=1e-4)


# --------------------------------------------------------------------------
# Sauvola adaptive thresholding
# --------------------------------------------------------------------------

def test_sauvola_threshold_values():
    g, _ = _synthetic_doc(60, 80)
    thresh = sauvola_threshold(g, window=25, k=0.15, R=128.0)
    assert thresh.shape == g.shape
    assert thresh.dtype == np.float64


def test_sauvola_binarize_mask():
    g, ink_gt = _synthetic_doc(100, 150)
    mask = sauvola_binarize(g, window=25, k=0.15, R=128.0)
    assert mask.dtype == bool
    assert mask.shape == g.shape
    # Under uneven illumination, Sauvola detects ink in both light and dark sides
    assert mask[ink_gt].mean() > 0.7


def test_sauvola_beats_global_otsu_on_uneven_light():
    g, ink_gt = _synthetic_doc(150, 250)
    otsu_mask = otsu_binarize(g)
    sauvola_mask = sauvola_binarize(g, window=25, k=0.15, R=128.0)

    # Compute F1 score for both
    def f1(pred, gt):
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom > 0 else 0.0

    f_otsu = f1(otsu_mask, ink_gt)
    f_sauv = f1(sauvola_mask, ink_gt)
    assert f_sauv > f_otsu
