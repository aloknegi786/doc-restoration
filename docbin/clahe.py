from __future__ import annotations

import numpy as np


def _build_lut(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    clip_px = max(1.0, clip_limit * tile.size / 256.0)
    excess = np.maximum(hist - clip_px, 0.0).sum()
    hist = np.minimum(hist, clip_px)
    hist += excess / 256.0

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
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("clahe expects a 2-D uint8 array")

    H, W = gray.shape
    gh, gw = grid_size

    luts = np.zeros((gh, gw, 256), dtype=np.uint8)
    for ity in range(gh):
        for itx in range(gw):
            y0 = int(round(ity * H / gh))
            y1 = int(round((ity + 1) * H / gh))
            x0 = int(round(itx * W / gw))
            x1 = int(round((itx + 1) * W / gw))
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0:
                luts[ity, itx] = np.arange(256, dtype=np.uint8)
            else:
                luts[ity, itx] = _build_lut(tile, clip_limit)

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)

    iy_frac = ys * gh / H - 0.5
    ix_frac = xs * gw / W - 0.5

    iy0 = np.clip(np.floor(iy_frac).astype(np.int32), 0, gh - 1)
    iy1 = np.clip(iy0 + 1, 0, gh - 1)
    ix0 = np.clip(np.floor(ix_frac).astype(np.int32), 0, gw - 1)
    ix1 = np.clip(ix0 + 1, 0, gw - 1)

    wy1 = np.clip(iy_frac - np.floor(iy_frac), 0.0, 1.0)
    wy0 = 1.0 - wy1
    wx1 = np.clip(ix_frac - np.floor(ix_frac), 0.0, 1.0)
    wx0 = 1.0 - wx1

    v = gray

    def lookup(iy_arr: np.ndarray, ix_arr: np.ndarray) -> np.ndarray:
        return luts[iy_arr[:, None], ix_arr[None, :], v].astype(np.float64)

    result = (
        wy0[:, None] * wx0[None, :] * lookup(iy0, ix0)
        + wy0[:, None] * wx1[None, :] * lookup(iy0, ix1)
        + wy1[:, None] * wx0[None, :] * lookup(iy1, ix0)
        + wy1[:, None] * wx1[None, :] * lookup(iy1, ix1)
    )
    return np.clip(np.round(result), 0, 255).astype(np.uint8)
