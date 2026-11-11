"""Estimate stroke width from an image, and derive a Sauvola window size
from it, replacing the window_from_relative() placeholder.

Method (classical, distance-transform based -- essentially a stroke width
transform restricted to the medial axis):
  1. Rough ink mask via Otsu on grayscale (just to find "roughly where the
     ink is"; it does not need to be a good final binarization).
  2. Skeletonize the mask, so every stroke collapses to its centreline.
  3. Distance-transform the mask: each foreground pixel's distance to the
     nearest background pixel.
  4. At skeleton pixels, distance * 2 is the local stroke thickness -- for
     a straight bar this is exact, because the skeleton runs along the
     centreline where the distance-to-edge is exactly half the width.
  5. Stroke width = median of these thicknesses (median, not mean, so a few
     thick blots or thin noise specks don't dominate).

The window size is then a small multiple of the stroke width, since Sauvola
needs to see a few strokes' worth of paper alongside the ink to build a
useful local mean/std. The multiplier ("factor") is not guessed by hand --
it is calibrated once against the frozen (window_rel, k) config from the
param sweep, in eval/calibrate_scale.py, on the tuning set only.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from docbin.segmentation import make_odd, otsu_binarize

DEFAULT_FACTOR = 10.0     # overwritten by scale_config.json once calibrated
DEFAULT_FLOOR = 15        # minimum window size in pixels, same floor as the placeholder rule
FALLBACK_STROKE_WIDTH = 4.0   # used only if no ink mask can be found at all


def estimate_stroke_width(gray: np.ndarray, min_ink_pixels: int = 200) -> float:
    """Median local stroke thickness in pixels. See module docstring."""
    ink = otsu_binarize(gray)
    if int(ink.sum()) < min_ink_pixels:
        return FALLBACK_STROKE_WIDTH

    skel = skeletonize(ink)
    if not skel.any():
        return FALLBACK_STROKE_WIDTH

    # cv2.distanceTransform wants nonzero = foreground, uint8.
    dist = cv2.distanceTransform((ink.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    widths = dist[skel] * 2.0
    widths = widths[widths > 0]
    if widths.size == 0:
        return FALLBACK_STROKE_WIDTH
    return float(np.median(widths))


def estimate_window(gray: np.ndarray, factor: float = DEFAULT_FACTOR,
                     floor: int = DEFAULT_FLOOR) -> int:
    """Sauvola window size derived from the image's own stroke width."""
    sw = estimate_stroke_width(gray)
    return make_odd(max(floor, factor * sw))
