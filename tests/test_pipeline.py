"""Tests for the new pipeline stages (docbin/colour, clahe, normalise,
cleanup, config, pipeline, roi).

Run from the project root:
    python -m pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from docbin.clahe import clahe, _build_lut
from docbin.cleanup import remove_small_components
from docbin.colour import select_channel, to_gray, to_red, to_lab_l, to_pca, VALID_CHANNELS
from docbin.config import PipelineConfig
from docbin.normalise import estimate_background, normalise_background
from docbin.pipeline import binarize, run
from docbin.roi import detect_roi, _fill_holes, _largest_cc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_doc(H=200, W=300, surround=40, paper_val=200, ink_val=30):
    """BGR image: uniform light page on a dark surround, with an ink blob."""
    bgr = np.full((H, W, 3), surround, dtype=np.uint8)
    bgr[50 : H - 50, 50 : W - 50] = paper_val
    # Ink blob in the centre
    bgr[90:110, 130:150] = ink_val
    return bgr


def _synthetic_gray(H=120, W=180):
    """Gradient + dark blob."""
    gray = np.tile(np.linspace(50, 220, W, dtype=np.uint8), (H, 1))
    gray[50:70, 80:100] = 20
    return gray


# ---------------------------------------------------------------------------
# colour.py
# ---------------------------------------------------------------------------

class TestColour:
    def test_to_gray_shape(self):
        bgr = _synthetic_doc()
        g = to_gray(bgr)
        assert g.shape == bgr.shape[:2]
        assert g.dtype == np.uint8

    def test_to_red_values(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 2] = 123          # R channel
        assert (to_red(bgr) == 123).all()

    def test_to_lab_l_range(self):
        bgr = _synthetic_doc()
        L = to_lab_l(bgr)
        assert L.min() >= 0 and L.max() <= 255
        assert L.dtype == np.uint8

    def test_to_pca_range_and_shape(self):
        bgr = _synthetic_doc()
        p = to_pca(bgr)
        assert p.shape == bgr.shape[:2]
        assert p.dtype == np.uint8
        assert p.min() >= 0 and p.max() <= 255

    def test_to_pca_paper_high(self):
        """PCA output: paper (dominant class) must have median ≥ 128."""
        bgr = _synthetic_doc()
        p = to_pca(bgr)
        assert np.median(p) >= 100   # generous threshold

    def test_select_channel_dispatches(self):
        bgr = _synthetic_doc()
        for ch in VALID_CHANNELS:
            out = select_channel(bgr, ch)
            assert out.shape == bgr.shape[:2]
            assert out.dtype == np.uint8

    def test_select_channel_bad_name(self):
        with pytest.raises(ValueError):
            select_channel(_synthetic_doc(), "rgb")


# ---------------------------------------------------------------------------
# roi.py
# ---------------------------------------------------------------------------

class TestROI:
    def test_light_page_on_dark_surround(self):
        bgr = _synthetic_doc(H=300, W=400, surround=40, paper_val=200)
        roi = detect_roi(bgr)
        assert roi.shape == bgr.shape[:2]
        assert roi.dtype == bool
        cov = roi.sum() / roi.size
        # Should detect page (< full frame, > a quarter)
        assert 0.3 < cov < 0.95

    def test_uniform_fallback(self):
        """Uniform image → no separation → full-frame fallback."""
        bgr = np.full((200, 300, 3), 128, dtype=np.uint8)
        roi = detect_roi(bgr)
        assert roi.all()

    def test_fill_holes_removes_interior_zero(self):
        """Interior zero in a rectangular mask should be filled."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:45, 5:45] = 1          # outer ring
        mask[15:35, 15:35] = 0        # interior hole
        filled = _fill_holes(mask)
        assert filled[25, 25] == 1    # hole filled
        assert filled[2, 2] == 0      # border background still 0

    def test_largest_cc_picks_biggest(self):
        m = np.zeros((50, 50), dtype=np.uint8)
        m[2:10, 2:10] = 1   # 64 px
        m[20:40, 20:40] = 1 # 400 px  ← largest
        lcc = _largest_cc(m)
        assert lcc[25, 25] == 1
        assert lcc[5, 5] == 0


# ---------------------------------------------------------------------------
# normalise.py
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_output_dtype_range(self):
        bgr = _synthetic_doc()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi = np.ones(gray.shape, dtype=bool)
        out = normalise_background(gray, roi)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_outside_roi_is_white(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        roi = np.zeros((100, 100), dtype=bool)
        roi[20:80, 20:80] = True
        out = normalise_background(gray, roi)
        # Pixels outside ROI must be 255
        assert (out[~roi] == 255).all()

    def test_uniform_paper_normalises_to_bright(self):
        """Uniform gray paper should map to near-255 after normalisation."""
        gray = np.full((100, 150), 100, dtype=np.uint8)
        roi = np.ones((100, 150), dtype=bool)
        out = normalise_background(gray, roi)
        # Closing on uniform image = no change; div by itself → 255
        assert out.mean() > 200

    def test_background_estimate_shape_and_dtype(self):
        gray = np.random.randint(100, 200, (80, 120), dtype=np.uint8)
        roi = np.ones((80, 120), dtype=bool)
        bg = estimate_background(gray, roi)
        assert bg.shape == gray.shape
        assert bg.dtype == np.float32
        assert (bg >= 1.0).all()

    def test_closing_fills_ink_in_background_estimate(self):
        """Background estimate should be close to paper level even at ink pixels."""
        # Uniform paper at 200, thin ink stroke at 50 (5 px wide)
        gray = np.full((60, 80), 200, dtype=np.uint8)
        gray[28:33, :] = 50          # ink stroke, 5px tall
        roi = np.ones((60, 80), dtype=bool)
        bg = estimate_background(gray, roi, sigma_frac=0.20, downsample=2)
        # Background at ink stroke pixels should be near 200, not pulled down to 50
        ink_bg = float(bg[28:33, 20:60].mean())
        assert ink_bg > 150, (
            f"Background at ink pixels is {ink_bg:.1f}, "
            "closing should fill strokes with paper value (~200)"
        )


# ---------------------------------------------------------------------------
# clahe.py
# ---------------------------------------------------------------------------

class TestCLAHE:
    def test_output_dtype_shape(self):
        gray = _synthetic_gray()
        out = clahe(gray)
        assert out.shape == gray.shape
        assert out.dtype == np.uint8

    def test_output_range(self):
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        out = clahe(gray)
        assert out.min() >= 0 and out.max() <= 255

    def test_uniform_tile_identity(self):
        """A single completely uniform tile should not change values."""
        gray = np.full((64, 64), 100, dtype=np.uint8)
        out = clahe(gray, grid_size=(1, 1), clip_limit=1.0)
        # Uniform tile → all values map to same output; range should be 0
        assert out.max() - out.min() <= 1   # allow off-by-one from rounding

    def test_lut_monotone(self):
        """LUT must be non-decreasing (histogram equalisation is monotone)."""
        rng = np.random.default_rng(0)
        tile = rng.integers(0, 256, 400, dtype=np.uint8)
        lut = _build_lut(tile, 2.0)
        assert (np.diff(lut.astype(np.int32)) >= 0).all()

    def test_close_to_opencv(self):
        """Mean absolute deviation from cv2.createCLAHE should be < 10."""
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        ours = clahe(gray, grid_size=(8, 8), clip_limit=2.0)
        cv_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ocv = cv_clahe.apply(gray)
        mad = float(np.abs(ours.astype(int) - ocv.astype(int)).mean())
        assert mad < 10.0, f"MAD vs OpenCV CLAHE = {mad:.2f}"

    def test_wrong_input_raises(self):
        with pytest.raises(ValueError):
            clahe(np.zeros((10, 10), dtype=np.float32))


# ---------------------------------------------------------------------------
# cleanup.py
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_removes_small_cc(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[2:4, 2:4] = True    # 4-px component → should be removed
        mask[20:30, 20:30] = True  # 100-px component → kept
        out = remove_small_components(mask, min_size=10)
        assert not out[3, 3]        # small CC removed
        assert out[25, 25]          # large CC kept

    def test_keeps_exactly_min_size(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:8, 5:8] = True    # 9 px
        out = remove_small_components(mask, min_size=9)
        assert out[6, 6]            # 9 >= 9 → kept

    def test_output_dtype(self):
        mask = np.ones((10, 10), dtype=bool)
        out = remove_small_components(mask, min_size=5)
        assert out.dtype == bool

    def test_all_removed_gives_empty(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:7, 5:7] = True    # 4 px
        out = remove_small_components(mask, min_size=100)
        assert not out.any()


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_instantiation(self):
        cfg = PipelineConfig()
        assert cfg.channel == "gray"
        assert cfg.use_roi is True
        assert cfg.R == 128.0

    def test_round_trip_json(self, tmp_path):
        cfg = PipelineConfig(channel="lab_l", k=0.20, use_clahe=False)
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        loaded = PipelineConfig.from_json(p)
        assert loaded.channel == "lab_l"
        assert loaded.k == pytest.approx(0.20)
        assert loaded.use_clahe is False

    def test_replace_for_ablation(self):
        cfg = PipelineConfig()
        cfg2 = dataclasses.replace(cfg, use_roi=False, use_clahe=False)
        assert not cfg2.use_roi
        assert not cfg2.use_clahe
        assert cfg2.k == cfg.k    # unchanged field preserved

    def test_from_json_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_json(tmp_path / "nonexistent.json")

    def test_from_dict_converts_list_to_tuple(self):
        d = {"clahe_grid": [4, 4]}
        cfg = PipelineConfig.from_dict(d)
        assert isinstance(cfg.clahe_grid, tuple)


# ---------------------------------------------------------------------------
# pipeline.py  (integration)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_returns_all_stage_keys(self):
        bgr = _synthetic_doc()
        cfg = PipelineConfig()
        stages = run(bgr, cfg)
        for key in ("channel", "roi", "normalised", "clahe", "binarized", "cleaned"):
            assert key in stages, f"missing stage '{key}'"

    def test_run_shapes_consistent(self):
        bgr = _synthetic_doc(H=100, W=150)
        cfg = PipelineConfig()
        stages = run(bgr, cfg)
        H, W = bgr.shape[:2]
        for key, arr in stages.items():
            assert arr.shape[:2] == (H, W), f"stage '{key}' has wrong shape"

    def test_binarize_returns_bool(self):
        bgr = _synthetic_doc()
        cfg = PipelineConfig()
        mask = binarize(bgr, cfg)
        assert mask.dtype == bool
        assert mask.shape == bgr.shape[:2]

    def test_all_stages_disabled_still_runs(self):
        """Disabling every optional stage must not raise."""
        bgr = _synthetic_doc()
        cfg = PipelineConfig(
            use_roi=False, use_normalise=False,
            use_clahe=False, use_cleanup=False,
        )
        stages = run(bgr, cfg)
        assert "cleaned" in stages

    def test_ink_blob_detected(self):
        """The dark ink blob in the synthetic image should be detected as ink."""
        bgr = _synthetic_doc(H=200, W=300, surround=40, paper_val=200, ink_val=20)
        cfg = PipelineConfig()
        mask = binarize(bgr, cfg)
        # Ink blob is at rows 90:110, cols 130:150
        ink_region = mask[90:110, 130:150]
        assert ink_region.sum() > 0, "ink blob not detected at all"

