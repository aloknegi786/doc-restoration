"""CLAHE — Contrast-Limited Adaptive Histogram Equalisation, from scratch.

Standard (OpenCV-compatible) tile-based algorithm:
  1. Divide the image into a grid_size[0] × grid_size[1] grid of tiles.
  2. For each tile compute the intensity histogram (256 bins).
  3. Clip the histogram at clip_limit × (tile_pixels / 256) and
     redistribute the excess uniformly across all bins.
  4. Build a CDF from the clipped histogram and normalise it to [0, 255]
     → one 256-entry look-up table (LUT) per tile.
  5. For every output pixel, bilinearly interpolate between the 4 nearest
     tile LUTs.  Border pixels are handled by clamping the tile indices
     (the nearest tile's LUT is used without extrapolation).

Validation: compare against cv2.createCLAHE() in tests/test_pipeline.py.
The two implementations will differ slightly at tile boundaries and may
differ in how fractional clip limits are rounded; mean absolute deviation
should be < 5 pixel values on a typical image.

Public API:
    clahe(gray, grid_size=(8, 8), clip_limit=2.0) -> np.ndarray  uint8
"""
from __future__ import annotations

import numpy as np


def _build_lut(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    """Build a 256-entry LUT for one tile with histogram clipping.

    Parameters
    ----------
    tile       : uint8 pixel values in this tile (any shape, flattened).
    clip_limit : clip threshold in units of *average frequency*.
                 clip_px = clip_limit * tile.size / 256.
    """
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    # Convert relative clip_limit to absolute pixel count
    clip_px = max(1.0, clip_limit * tile.size / 256.0)
    excess = np.maximum(hist - clip_px, 0.0).sum()
    hist = np.minimum(hist, clip_px)
    hist += excess / 256.0            # redistribute excess uniformly

    cdf = np.cumsum(hist)
    cdf_min = cdf[0]
    cdf_max = cdf[-1]
    if cdf_max > cdf_min:
        lut = np.round((cdf - cdf_min) / (cdf_max - cdf_min) * 255.0)
    else:
        lut = np.arange(256, dtype=np.float64)
    return lut.astype(np.uint8)


def clahe(
    gray: np.ndarray,
    grid_size: tuple[int, int] = (8, 8),
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image.

    Parameters
    ----------
    gray       : uint8 grayscale image (H, W).
    grid_size  : (num_rows, num_cols) of tiles.
    clip_limit : contrast-limiting factor (relative to average frequency).
                 2.0 is a mild enhancement; higher values amplify texture.

    Returns
    -------
    uint8 image (H, W) with contrast enhanced locally.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("clahe expects a 2-D uint8 array")

    H, W = gray.shape
    gh, gw = grid_size                # number of tiles vertically / horizontally

    # Build a LUT for each tile ------------------------------------------
    luts = np.zeros((gh, gw, 256), dtype=np.uint8)
    for ity in range(gh):
        for itx in range(gw):
            # Tile boundaries (last tile may be slightly larger)
            y0 = int(round(ity * H / gh))
            y1 = int(round((ity + 1) * H / gh))
            x0 = int(round(itx * W / gw))
            x1 = int(round((itx + 1) * W / gw))
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0:
                luts[ity, itx] = np.arange(256, dtype=np.uint8)
            else:
                luts[ity, itx] = _build_lut(tile, clip_limit)

    # Bilinear interpolation using all 4 surrounding tile LUTs -----------
    # Map each pixel to a fractional tile-centre coordinate.
    # The centre of tile (ity, itx) is at:
    #   cy = (ity + 0.5) * H / gh
    #   cx = (itx + 0.5) * W / gw
    # So the fractional tile row for pixel y is:  y * gh / H - 0.5

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)

    iy_frac = ys * gh / H - 0.5    # (H,)
    ix_frac = xs * gw / W - 0.5    # (W,)

    iy0 = np.clip(np.floor(iy_frac).astype(np.int32), 0, gh - 1)   # (H,)
    iy1 = np.clip(iy0 + 1,                              0, gh - 1)
    ix0 = np.clip(np.floor(ix_frac).astype(np.int32), 0, gw - 1)   # (W,)
    ix1 = np.clip(ix0 + 1,                              0, gw - 1)

    wy1 = np.clip(iy_frac - np.floor(iy_frac), 0.0, 1.0)   # (H,)
    wy0 = 1.0 - wy1
    wx1 = np.clip(ix_frac - np.floor(ix_frac), 0.0, 1.0)   # (W,)
    wx0 = 1.0 - wx1

    v = gray   # (H, W) pixel values, used as LUT indices

    def lookup(iy_arr: np.ndarray, ix_arr: np.ndarray) -> np.ndarray:
        """Vectorised LUT lookup.  Returns float64 (H, W)."""
        return luts[iy_arr[:, None], ix_arr[None, :], v].astype(np.float64)

    result = (
        wy0[:, None] * wx0[None, :] * lookup(iy0, ix0)
        + wy0[:, None] * wx1[None, :] * lookup(iy0, ix1)
        + wy1[:, None] * wx0[None, :] * lookup(iy1, ix0)
        + wy1[:, None] * wx1[None, :] * lookup(iy1, ix1)
    )
    return np.clip(np.round(result), 0, 255).astype(np.uint8)

