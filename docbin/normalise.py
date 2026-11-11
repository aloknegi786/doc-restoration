"""Background normalisation via division.

Uneven illumination (shadows, stains, fading) makes a fixed or even a local
threshold struggle because the background level varies across the image.
Division normalisation corrects this:

    normalised(x,y) = clip(255 * gray(x,y) / background(x,y), 0, 255)

After this, paper pixels are near 255 everywhere, and ink pixels are
proportionally darker, regardless of the original illumination gradient.

Background estimation uses **morphological closing** on a downsampled copy
of the image.  Closing = dilation (max-filter) followed by erosion; on a
document image this correctly fills thin dark ink strokes with the
surrounding bright paper values, giving a pure-paper background estimate
uncontaminated by the ink.  A Gaussian blur would average ink and paper
together, underestimating the background and distorting the normalised
image in ink-dense regions — that is why closing is used here.

Outside the ROI, pixels are set to white (255) before downsampling so the
dark surround does not bleed into the background estimate near the document
border.  After normalisation, pixels outside the ROI are set to 255 so that
Sauvola classifies the surround correctly as non-ink.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(
    gray: np.ndarray,
    roi: np.ndarray,
    *,
    sigma_frac: float = 0.10,
    downsample: int = 4,
) -> np.ndarray:
    """Estimate the spatially-varying background level (paper colour).

    Uses morphological closing on a downsampled image so that thin ink
    strokes (dark) are replaced by the surrounding paper brightness (light).
    The closing kernel radius is proportional to `sigma_frac * short_side`.

    Parameters
    ----------
    gray        : uint8 grayscale image (H, W).
    roi         : bool mask (H, W) — True = document region.
    sigma_frac  : closing kernel radius as a fraction of the shorter image
                  side.  Larger → more aggressive ink-stroke filling.
    downsample  : downscale factor applied before closing (speed).
                  At 4× a 5-px stroke becomes ~1 px and is reliably filled.

    Returns
    -------
    float32 background estimate (H, W).  Values < 1 are clamped to 1 to
    prevent division by zero in callers.
    """
    H, W = gray.shape
    radius_full = max(10.0, sigma_frac * min(H, W))

    # Set surround (outside ROI) to white before downsampling, so it does
    # not contaminate the background estimate near the document edge.
    work = gray.copy()
    work[~roi] = 255

    # Downsample for speed
    h_s = max(1, H // downsample)
    w_s = max(1, W // downsample)
    small = cv2.resize(work, (w_s, h_s), interpolation=cv2.INTER_AREA)

    # Morphological closing: fills ink strokes with surrounding paper values.
    radius_s = max(3, int(radius_full / downsample))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius_s * 2 + 1, radius_s * 2 + 1)
    )
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)

    # Upsample back to original resolution with bilinear interpolation
    # (smooth transitions, no blocking artefacts).
    bg = cv2.resize(closed.astype(np.float32), (W, H),
                    interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 1.0).astype(np.float32)


def normalise_background(
    gray: np.ndarray,
    roi: np.ndarray,
    *,
    sigma_frac: float = 0.10,
    downsample: int = 4,
) -> np.ndarray:
    """Return a background-normalised uint8 image.

    Inside the ROI  : paper → ~255, ink proportionally darker.
    Outside the ROI : set to 255 (pure white background) so Sauvola
                      classifies the surround correctly as non-ink.

    Parameters
    ----------
    gray       : uint8 grayscale image (H, W).
    roi        : bool mask (H, W) — True = document region.
    sigma_frac : see `estimate_background`.
    downsample : see `estimate_background`.

    Returns
    -------
    uint8 normalised image (H, W).
    """
    bg = estimate_background(gray, roi, sigma_frac=sigma_frac, downsample=downsample)
    normalised = np.clip(255.0 * gray.astype(np.float32) / bg, 0.0, 255.0)
    result = normalised.astype(np.uint8)
    # Pixels outside the ROI → pure white (background class)
    result[~roi] = 255
    return result

