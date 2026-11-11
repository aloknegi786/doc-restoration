import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover, load_sample
from docbin.segmentation import (
    local_mean_std, niblack_threshold, otsu_threshold, sauvola_threshold,
)
from eval.metrics import compute_metrics


def synthetic_doc(h=200, w=300, seed=0):
    """Dark strokes on paper with a left-to-right illumination gradient."""
    rng = np.random.default_rng(seed)
    bg = np.tile(np.linspace(120, 230, w), (h, 1))
    img = bg.copy()
    ink = np.zeros((h, w), bool)
    for y in range(30, h - 20, 40):
        ink[y:y + 3, 20:w - 20] = True
        ink[y - 8:y + 8, 60:63] = True
    img[ink] *= 0.35
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), ink


# ---------------- Otsu ----------------
def test_otsu_matches_skimage():
    from skimage.filters import threshold_otsu
    g, _ = synthetic_doc()
    assert abs(otsu_threshold(g) - int(threshold_otsu(g))) <= 1


def test_otsu_bimodal_exact():
    g = np.concatenate([np.full(500, 50), np.full(500, 200)]).astype(np.uint8).reshape(20, 50)
    t = otsu_threshold(g)
    assert 50 <= t < 200


# ---------------- integral-image statistics ----------------
def test_local_mean_std_matches_bruteforce():
    g, _ = synthetic_doc(60, 80)
    w = 11
    m, s = local_mean_std(g, w)
    pad = w // 2
    gp = np.pad(g.astype(float), pad, mode="reflect")
    for (y, x) in [(0, 0), (5, 70), (30, 40), (59, 79)]:
        win = gp[y:y + w, x:x + w]
        assert m[y, x] == pytest.approx(win.mean(), abs=1e-6)
        assert s[y, x] == pytest.approx(win.std(), abs=1e-4)


def test_sauvola_matches_skimage():
    from skimage.filters import threshold_sauvola
    g, _ = synthetic_doc()
    mine = sauvola_threshold(g, window=25, k=0.2, R=128.0)
    ref = threshold_sauvola(g, window_size=25, k=0.2, r=128.0)
    # skimage may use a slightly different std normalisation, so compare
    # the resulting binary masks, which is what actually matters.
    agree = np.mean((g <= mine) == (g <= ref))
    assert agree > 0.995, agree


def test_niblack_matches_skimage():
    from skimage.filters import threshold_niblack
    g, _ = synthetic_doc()
    # Sign convention differs: original Niblack is T = m + k*s (k negative),
    # skimage uses T = m - k*s. So my k=-0.2 equals skimage k=+0.2.
    mine = niblack_threshold(g, window=25, k=-0.2)
    ref = threshold_niblack(g, window_size=25, k=0.2)
    assert np.abs(mine - ref).max() < 1e-6


def test_sauvola_beats_otsu_on_uneven_light():
    from eval.metrics import compute_metrics as cm
    from docbin.segmentation import otsu_binarize, sauvola_binarize
    g, ink = synthetic_doc()
    f_otsu = cm(otsu_binarize(g), ink)["f_measure"]
    f_sauv = cm(sauvola_binarize(g, 25, 0.2), ink)["f_measure"]
    assert f_sauv > f_otsu


# ---------------- metrics ----------------
def test_metrics_perfect_and_worst():
    gt = np.zeros((10, 10), bool)
    gt[2:5, 2:5] = True
    m = compute_metrics(gt.copy(), gt)
    assert m["f_measure"] == 1.0 and m["dice"] == 1.0 and m["psnr"] == float("inf")
    m = compute_metrics(~gt, gt)
    assert m["f_measure"] == 0.0


def test_f_equals_dice_and_hand_example():
    gt = np.zeros((10, 10), bool)
    pred = np.zeros((10, 10), bool)
    gt[0, :4] = True          # 4 ink pixels
    pred[0, 2:6] = True       # 2 TP, 2 FP, 2 FN
    m = compute_metrics(pred, gt)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["f_measure"] == pytest.approx(0.5)
    assert m["dice"] == pytest.approx(m["f_measure"])
    assert m["psnr"] == pytest.approx(10 * np.log10(1 / (4 / 100)))


def test_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((3, 3), bool), np.zeros((4, 4), bool))


# ---------------- loader ----------------
def test_discover_dibco_and_palm(tmp_path):
    g, ink = synthetic_doc(50, 60)
    gt_img = np.where(ink, 0, 255).astype(np.uint8)

    d = tmp_path / "dibco2018"
    d.mkdir()
    cv2.imwrite(str(d / "1.bmp"), g)
    cv2.imwrite(str(d / "1_gt.bmp"), gt_img)
    cv2.imwrite(str(d / "2.bmp"), g)                      # no GT -> skipped

    p = tmp_path / "palm"
    p.mkdir()
    cv2.imwrite(str(p / "Bangli-P9.jpg"), g)
    cv2.imwrite(str(p / "Bangli-P9_GT1.bmp"), gt_img)
    cv2.imwrite(str(p / "Bangli-P9_GT2.bmp"), gt_img)

    s1 = discover(d, "dibco2018")
    assert [s.id for s in s1] == ["dibco2018/1"]
    assert len(s1[0].gt_paths) == 1

    s2 = discover(p, "palm")
    assert [s.id for s in s2] == ["palm/Bangli-P9"]
    assert len(s2[0].gt_paths) == 2

    img, gts = load_sample(s1[0])
    assert gts[0].dtype == bool and gts[0].sum() == ink.sum()


def test_load_sample_dimension_mismatch(tmp_path):
    g, ink = synthetic_doc(50, 60)
    cv2.imwrite(str(tmp_path / "1.bmp"), g)
    cv2.imwrite(str(tmp_path / "1_gt.bmp"), np.full((40, 40), 255, np.uint8))
    (s,) = discover(tmp_path, "x")
    with pytest.raises(ValueError):
        load_sample(s)


def test_discover_separated_layout(tmp_path):
    g, ink = synthetic_doc(50, 60)
    gt_img = np.where(ink, 0, 255).astype(np.uint8)

    # DIBCO-style: images/ + gt/ with N_gt.bmp
    d = tmp_path / "dibco2018"
    (d / "images").mkdir(parents=True); (d / "gt").mkdir()
    cv2.imwrite(str(d / "images" / "1.bmp"), g)
    cv2.imwrite(str(d / "gt" / "1_gt.bmp"), gt_img)
    cv2.imwrite(str(d / "images" / "2.bmp"), g)          # no GT -> skipped
    (s,) = discover(d, "dibco2018")
    assert s.id == "dibco2018/1" and len(s.gt_paths) == 1

    # palm-style: images/ + gt1/ + gt2/
    p = tmp_path / "palm"
    for sub in ("images", "gt1", "gt2"):
        (p / sub).mkdir(parents=True)
    cv2.imwrite(str(p / "images" / "WN-P1a.jpg"), g)
    cv2.imwrite(str(p / "gt1" / "WN-P1a_GT1.bmp"), gt_img)
    cv2.imwrite(str(p / "gt2" / "WN-P1a_GT2.bmp"), gt_img)
    cv2.imwrite(str(p / "images" / "Only1.jpg"), g)      # only GT1 -> kept, warned
    cv2.imwrite(str(p / "gt1" / "Only1_GT1.bmp"), gt_img)
    ss = {x.id: x for x in discover(p, "palm")}
    assert len(ss["palm/WN-P1a"].gt_paths) == 2
    assert len(ss["palm/Only1"].gt_paths) == 1
    img, gts = load_sample(ss["palm/WN-P1a"])
    assert len(gts) == 2 and gts[0].sum() == ink.sum()


# ---------------- param_sweep refinement logic ----------------
def test_refine_picks_worst_case_not_mean():
    from eval.param_sweep import refined_grid_points, frange

    fold_a = {"window_rel": 0.06, "k": 0.15}
    fold_b = {"window_rel": 0.08, "k": 0.30}
    rels, ks = refined_grid_points(fold_a, fold_b)
    assert min(rels) <= 0.05 and max(rels) >= 0.09
    assert min(ks) <= 0.13 and max(ks) >= 0.32
    assert rels == sorted(rels) and ks == sorted(ks)

    assert frange(0.05, 0.09, 0.01) == [0.05, 0.06, 0.07, 0.08, 0.09]


def test_worst_case_selection_beats_mean_on_lopsided_point():
    # Point P3 has a high peak on one year and a low trough on the other;
    # P2 is steadier. Worst-case must prefer P2 even though P3 could win on mean.
    scored = [
        {"point": (0.05, 0.10), "f_2018": 0.72, "f_2019": 0.58, "mean": 0.65, "min": 0.58},
        {"point": (0.06, 0.15), "f_2018": 0.68, "f_2019": 0.66, "mean": 0.67, "min": 0.66},
        {"point": (0.08, 0.30), "f_2018": 0.75, "f_2019": 0.52, "mean": 0.635, "min": 0.52},
    ]
    best_worst_case = max(scored, key=lambda r: r["min"])
    best_mean = max(scored, key=lambda r: r["mean"])
    assert best_worst_case["point"] == (0.06, 0.15)
    assert best_mean["point"] == (0.06, 0.15)  # in this example they coincide
    # a genuinely lopsided case where they diverge
    scored[2]["f_2018"], scored[2]["mean"] = 0.95, (0.95 + 0.52) / 2
    scored[2]["min"] = min(0.95, 0.52)
    best_worst_case = max(scored, key=lambda r: r["min"])
    best_mean = max(scored, key=lambda r: r["mean"])
    assert best_worst_case["point"] == (0.06, 0.15)
    assert best_mean["point"] == (0.08, 0.30)
    assert best_worst_case["point"] != best_mean["point"]


# ---------------- stroke-width estimation ----------------
def test_stroke_width_matches_known_bar_thickness():
    from docbin.scale import estimate_stroke_width

    for thickness in (3, 5, 9, 15):
        img = np.full((300, 300), 230, np.uint8)
        # several long, straight, well-separated horizontal bars of a known
        # thickness -- skeleton = centreline, distance*2 = exact thickness
        for y in range(30, 280, 40):
            img[y:y + thickness, 20:280] = 20
        sw = estimate_stroke_width(img)
        assert abs(sw - thickness) <= 1.0, (thickness, sw)


def test_stroke_width_blank_image_falls_back():
    from docbin.scale import FALLBACK_STROKE_WIDTH, estimate_stroke_width
    blank = np.full((100, 100), 230, np.uint8)
    assert estimate_stroke_width(blank) == FALLBACK_STROKE_WIDTH


def test_estimate_window_scales_with_stroke_width():
    from docbin.scale import estimate_window

    def bars(thickness):
        img = np.full((300, 300), 230, np.uint8)
        for y in range(30, 280, 40):
            img[y:y + thickness, 20:280] = 20
        return img

    w_thin = estimate_window(bars(3), factor=10.0, floor=1)
    w_thick = estimate_window(bars(9), factor=10.0, floor=1)
    assert w_thick > w_thin
    assert w_thin % 2 == 1 and w_thick % 2 == 1     # always odd


def test_estimate_window_respects_floor():
    from docbin.scale import estimate_window
    img = np.full((100, 100), 230, np.uint8)
    img[50:51, 10:90] = 20            # a 1px hairline -> tiny stroke width
    w = estimate_window(img, factor=1.0, floor=25)
    assert w >= 25


# ---------------- diagnostics ----------------
def test_ink_fraction():
    from eval.diagnostics import ink_fraction
    m = np.zeros((10, 10), bool)
    m[:2, :] = True
    assert ink_fraction(m) == pytest.approx(0.2)


def test_paired_wilcoxon_detects_consistent_gain_and_null():
    from eval.diagnostics import paired_wilcoxon
    rng = np.random.default_rng(0)
    real = rng.uniform(0.01, 0.03, 15)                 # all positive: clear effect
    res = paired_wilcoxon(real)
    assert res["wins"] == 15 and res["p"] < 0.01
    null = np.array([0.02, -0.02, 0.01, -0.01, 0.03, -0.03, 0.015, -0.015])   # symmetric around 0
    assert paired_wilcoxon(null)["p"] > 0.5


def test_paired_wilcoxon_too_few_pairs_returns_nan():
    from eval.diagnostics import paired_wilcoxon
    res = paired_wilcoxon(np.array([0.0, 0.0, 0.01, 0.0]))
    assert np.isnan(res["p"])
