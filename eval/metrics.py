"""Evaluation metrics. Inputs are boolean arrays with INK = True.

F-Measure and Dice are mathematically identical for binary masks
(both equal 2TP / (2TP + FP + FN)); both are returned because the project
brief names both.
"""
from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Return precision, recall, f_measure, dice, psnr (higher = better)."""
    tp, fp, fn, _ = confusion(pred, gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom > 0 else 1.0    # both empty -> perfect

    mse = (fp + fn) / pred.size                      # binary images, C = 1
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "dice": dice,
        "psnr": psnr,
    }
