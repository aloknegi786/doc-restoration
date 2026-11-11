"""Calibrate the stroke-width -> window-size factor, and sanity-check it.

The param sweep froze a pixel window rule (window_from_relative, i.e. a
fraction of the shorter image side) together with k. That rule doesn't
transfer across very different image sizes and substrates (see the palm
leaf vs DIBCO discussion). estimate_window() in docbin/scale.py replaces it
with a rule based on the image's own stroke width, but the multiplier
("factor") has to come from data, not a guess.

This script, on the DIBCO 2018+2019 TUNING set only:
  1. For every tuning image, computes the window the frozen config would
     have used (window_from_relative with the frozen window_rel), and the
     image's estimated stroke width.
  2. factor_i = frozen_window_px_i / stroke_width_i, for each image.
  3. Reports the median factor (used as the calibrated value) and its
     spread (how much it varies image to image -- high spread would mean
     stroke width doesn't explain the frozen window well).
  4. Sanity check: re-scores Sauvola (same frozen k) using
     estimate_window(factor) instead of the fixed relative window, on both
     years separately, and compares F-Measure against the frozen
     window_from_relative baseline. This is NOT re-tuning k or factor
     against these scores -- factor is fixed from step 2/3 before this
     check runs, and the check is reported, not optimised.

Never touches dibco2017 or palm.

Usage:
    python -m eval.calibrate_scale --dibco2018 data/dibco2018 --dibco2019 data/dibco2019 \
        --splits data/splits.json --sauvola-config eval/results/sauvola_config.json \
        --out eval/results/scale_calibration.csv --config-out eval/results/scale_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover, load_sample, to_gray
from docbin.scale import DEFAULT_FLOOR, estimate_stroke_width, estimate_window
from docbin.segmentation import sauvola_binarize, window_from_relative
from eval.metrics import compute_metrics


def load_year(root, dataset, ids):
    wanted = set(ids)
    out = []
    for s in discover(root, dataset):
        if s.id not in wanted:
            continue
        bgr, gts = load_sample(s)
        out.append((s.id, to_gray(bgr), gts))
    missing = wanted - {i for i, _, _ in out}
    if missing:
        raise SystemExit(f"error: splits.json expects {sorted(missing)} from {dataset}, not found on disk")
    return out


def mean_f(pred, gts) -> float:
    return float(np.mean([compute_metrics(pred, g)["f_measure"] for g in gts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dibco2018", required=True)
    ap.add_argument("--dibco2019", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--sauvola-config", default="eval/results/sauvola_config.json",
                     help="frozen (window_rel, k) from param_sweep.py")
    ap.add_argument("--out", default="eval/results/scale_calibration.csv")
    ap.add_argument("--config-out", default="eval/results/scale_config.json")
    args = ap.parse_args()

    sauvola_cfg = json.loads(Path(args.sauvola_config).read_text())
    window_rel = sauvola_cfg["window_rel"]
    k = sauvola_cfg["k"]
    print(f"frozen config: window_rel={window_rel}, k={k}")

    splits = json.loads(Path(args.splits).read_text())
    ids18, ids19 = splits["tuning"]["dibco2018"], splits["tuning"]["dibco2019"]
    if not ids18 or not ids19:
        raise SystemExit("error: splits.json has an empty tuning set for 2018 or 2019")

    imgs18 = load_year(args.dibco2018, "dibco2018", ids18)
    imgs19 = load_year(args.dibco2019, "dibco2019", ids19)
    all_imgs = imgs18 + imgs19

    # -------- step 1-3: per-image factor --------
    rows = []
    for img_id, gray, gts in all_imgs:
        frozen_win = window_from_relative(gray.shape, window_rel)
        sw = estimate_stroke_width(gray)
        factor = frozen_win / sw if sw > 0 else float("nan")
        rows.append({"id": img_id, "dataset": img_id.split("/")[0], "stroke_width": sw,
                     "frozen_window_px": frozen_win, "factor": factor})

    factors = np.array([r["factor"] for r in rows if np.isfinite(r["factor"])])
    calibrated_factor = float(np.median(factors))
    print(f"\nper-image factor: median={calibrated_factor:.2f}, "
          f"mean={factors.mean():.2f}, std={factors.std():.2f}, "
          f"min={factors.min():.2f}, max={factors.max():.2f}")
    if factors.std() / factors.mean() > 0.5:
        print("  [note] high relative spread -- stroke width alone does not fully "
              "explain the frozen window across these images; treat the factor as "
              "a reasonable default, not a precise law, and say so in the report.")

    # -------- step 4: sanity check (report only, not optimised) --------
    def score_year(imgs, use_estimate: bool) -> float:
        fs = []
        for _, gray, gts in imgs:
            win = (estimate_window(gray, factor=calibrated_factor, floor=DEFAULT_FLOOR)
                   if use_estimate else window_from_relative(gray.shape, window_rel))
            pred = sauvola_binarize(gray, window=win, k=k)
            fs.append(mean_f(pred, gts))
        return float(np.mean(fs))

    print("\nsanity check (frozen k, same for both):")
    for name, imgs in (("2018", imgs18), ("2019", imgs19)):
        f_fixed = score_year(imgs, use_estimate=False)
        f_stroke = score_year(imgs, use_estimate=True)
        delta = f_stroke - f_fixed
        flag = "" if delta >= -0.01 else "  [note] stroke-width rule is worse here"
        print(f"  {name}: window_from_relative F={f_fixed:.4f}   "
              f"estimate_window F={f_stroke:.4f}   delta={delta:+.4f}{flag}")

    # -------- write outputs --------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "stroke_width", "frozen_window_px", "factor"])
        w.writeheader()
        w.writerows(rows)

    config = {
        "factor": calibrated_factor,
        "floor": DEFAULT_FLOOR,
        "calibrated_from": "median(frozen_window_px / stroke_width) over dibco2018+2019 tuning set",
        "factor_spread_std": float(factors.std()),
        "factor_spread_mean": float(factors.mean()),
        "note": "This factor is a default, not re-tuned against 2017 or palm. "
                "If it performs poorly on those, that is a reported finding, "
                "not a reason to adjust it after seeing their scores.",
    }
    Path(args.config_out).write_text(json.dumps(config, indent=2))
    print(f"\nwrote per-image table -> {out}")
    print(f"wrote scale config -> {args.config_out}  (factor={calibrated_factor:.2f})")


if __name__ == "__main__":
    main()
