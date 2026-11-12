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
    stages: dict[str, np.ndarray] = {}

    gray = select_channel(bgr, cfg.channel)
    stages["channel"] = gray

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

    if cfg.use_normalise:
        gray = normalise_background(
            gray,
            roi,
            sigma_frac=cfg.bg_sigma_frac,
            downsample=cfg.bg_downsample,
        )
    stages["normalised"] = gray

    if cfg.use_clahe:
        gray = clahe(gray, grid_size=cfg.clahe_grid, clip_limit=cfg.clahe_clip)
    stages["clahe"] = gray

    window = window_from_relative(gray.shape, cfg.window_rel)
    mask = sauvola_binarize(gray, window=window, k=cfg.k, R=cfg.R)
    stages["binarized"] = mask

    if cfg.use_cleanup:
        cleaned = remove_small_components(mask, min_size=cfg.min_component_size)
    else:
        cleaned = mask
    stages["cleaned"] = cleaned

    return stages


def binarize(bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    return run(bgr, cfg)["cleaned"]
