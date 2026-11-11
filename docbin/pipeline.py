"""End-to-end document binarization pipeline.

Acquisition → channel selection → ROI detection → background normalisation
→ CLAHE → Sauvola binarization → small-component cleanup

Every stage is a pure function (ndarray + params → ndarray).  The pipeline
returns a dict of *intermediates* (one entry per stage), which is needed
for:
  * the FastAPI /binarize/stages route (returns all stages as images)
  * ablation.py (compare each stage's output with and without it)
  * failure-analysis figures in the report

Public API:
    run(bgr, cfg)        -> dict[str, np.ndarray]   (all intermediates)
    binarize(bgr, cfg)   -> np.ndarray[bool]         (final mask only)

Stage keys in the returned dict:
    "channel"     uint8 (H,W)   after channel selection
    "roi"         bool  (H,W)   detected document region
    "normalised"  uint8 (H,W)   after background normalisation
    "clahe"       uint8 (H,W)   after CLAHE
    "binarized"   bool  (H,W)   after Sauvola
    "cleaned"     bool  (H,W)   after small-component removal  ← final output
"""
from __future__ import annotations

import numpy as np

from docbin.clahe import clahe
from docbin.cleanup import remove_small_components
from docbin.colour import select_channel
from docbin.config import PipelineConfig
from docbin.normalise import normalise_background
from docbin.roi import detect_roi
from docbin.segmentation import sauvola_binarize, window_from_relative


def run(bgr: np.ndarray, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    """Run the full pipeline and return all intermediates.

    Parameters
    ----------
    bgr : uint8 BGR colour image (H, W, 3).
    cfg : PipelineConfig with all stage parameters.

    Returns
    -------
    dict with keys: 'channel', 'roi', 'normalised', 'clahe',
                    'binarized', 'cleaned'.
    """
    stages: dict[str, np.ndarray] = {}

    # 1. Channel selection --------------------------------------------------
    gray = select_channel(bgr, cfg.channel)
    stages["channel"] = gray

    # 2. ROI detection ------------------------------------------------------
    if cfg.use_roi:
        roi = detect_roi(
            bgr,
            margin_frac=cfg.roi_margin_frac,
            close_frac=cfg.roi_close_frac,
            downsample=cfg.roi_downsample,
        )
    else:
        roi = np.ones(gray.shape, dtype=bool)
    stages["roi"] = roi

    # 3. Background normalisation -------------------------------------------
    if cfg.use_normalise:
        gray = normalise_background(
            gray, roi,
            sigma_frac=cfg.bg_sigma_frac,
            downsample=cfg.bg_downsample,
        )
    stages["normalised"] = gray

    # 4. CLAHE --------------------------------------------------------------
    if cfg.use_clahe:
        gray = clahe(gray, grid_size=cfg.clahe_grid, clip_limit=cfg.clahe_clip)
    stages["clahe"] = gray

    # 5. Sauvola binarization -----------------------------------------------
    window = window_from_relative(gray.shape, cfg.window_rel)
    mask = sauvola_binarize(gray, window=window, k=cfg.k, R=cfg.R)
    stages["binarized"] = mask

    # 6. Small-component cleanup --------------------------------------------
    if cfg.use_cleanup:
        cleaned = remove_small_components(mask, min_size=cfg.min_component_size)
    else:
        cleaned = mask
    stages["cleaned"] = cleaned

    return stages


def binarize(bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Convenience wrapper: run the pipeline and return only the final mask.

    Parameters
    ----------
    bgr : uint8 BGR colour image (H, W, 3).
    cfg : PipelineConfig.

    Returns
    -------
    Boolean ink mask (H, W) — True = ink, False = background.
    """
    return run(bgr, cfg)["cleaned"]

