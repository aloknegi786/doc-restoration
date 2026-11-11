"""Diagnostics on the baseline results. No tuning, no new parameters.

Two questions, both raised by the baseline_v2 results:

1. WHY did the stroke-width window regress on palm leaf?
   Hypothesis: estimate_stroke_width() seeds itself with a rough Otsu mask,
   and on palm leaf that mask is mostly leaf, not ink. Test: per image,
   compare the fraction of pixels the Otsu seed marks as ink against the
   fraction the ground truth marks as ink. If the hypothesis holds, the
   ratio (seed / GT) should be near 1-3x on DIBCO and far larger on palm.

2. Is the sauvola_stroke vs sauvola_rel difference real?
   Paired Wilcoxon signed-rank test on the per-image F-Measure deltas, per
   dataset. With 10-20 images per set, a mean gain of 0.01-0.02 is only
   evidence if the test says so.

Usage:
    python -m eval.diagnostics \
        --dataset dibco2018=data/dibco2018 --dataset dibco2019=data/dibco2019 \
        --dataset dibco2017=data/dibco2017 --dataset palm=data/palm_split \
        --splits data/splits.json --baseline-csv eval/results/baseline_v2.csv \
        --out eval/results/diagnostics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover, load_sample, to_gray
from docbin.scale import estimate_stroke_width
from docbin.segmentation import otsu_binarize


def ink_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def paired_wilcoxon(deltas: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences."""
    deltas = np.asarray(deltas, dtype=float)
    n_nonzero = int(np.count_nonzero(deltas))
    out = {"n": len(deltas), "mean": float(deltas.mean()), "median": float(np.median(deltas)),
           "wins": int((deltas > 0).sum()), "losses": int((deltas < 0).sum()), "p": float("nan")}
    if n_nonzero < 6:           # too few informative pairs for a meaningful test
        return out
    out["p"] = float(stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue)
    return out


def per_image_diagnostics(args) -> list[dict]:
    excluded: set[str] = set()
    if args.splits:
        excluded = set(json.loads(Path(args.splits).read_text()).get("palm_dev", []))

    rows = []
    for spec in args.dataset:
        name, _, path = spec.partition("=")
        for s in discover(path, name):
            if s.id in excluded:
                continue
            bgr, gts = load_sample(s)
            gray = to_gray(bgr)
            seed = otsu_binarize(gray)
            seed_frac = ink_fraction(seed)
            gt_frac = float(np.mean([ink_fraction(g) for g in gts]))
            rows.append({
                "id": s.id, "dataset": name,
                "seed_ink_fraction": seed_frac,
                "gt_ink_fraction": gt_frac,
                "seed_to_gt_ratio": seed_frac / gt_frac if gt_frac > 0 else float("nan"),
                "stroke_width": estimate_stroke_width(gray),
            })
        print(f"[{name}] diagnostics computed")
    return rows


def load_deltas(baseline_csv: str) -> dict[str, dict[str, float]]:
    """{dataset: {image_id: F(sauvola_stroke) - F(sauvola_rel)}} from run_baseline output."""
    f: dict[tuple[str, str], dict[str, float]] = {}
    with open(baseline_csv) as fh:
        for r in csv.DictReader(fh):
            if r["method"] in ("sauvola_rel", "sauvola_stroke"):
                f.setdefault((r["dataset"], r["id"]), {})[r["method"]] = float(r["f_measure"])
    out: dict[str, dict[str, float]] = {}
    for (ds, img_id), m in f.items():
        if "sauvola_rel" in m and "sauvola_stroke" in m:
            out.setdefault(ds, {})[img_id] = m["sauvola_stroke"] - m["sauvola_rel"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--splits")
    ap.add_argument("--baseline-csv", default="eval/results/baseline_v2.csv")
    ap.add_argument("--out", default="eval/results/diagnostics.csv")
    args = ap.parse_args()

    # Check for the baseline CSV *before* the slow per-image loop, so a
    # missing file fails fast rather than after minutes of computation.
    baseline_path = Path(args.baseline_csv)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"baseline CSV not found: {baseline_path}\n"
            "Run eval/run_baseline.py first."
        )

    rows = per_image_diagnostics(args)
    deltas = load_deltas(args.baseline_csv)
    for r in rows:
        r["delta_f_stroke_minus_rel"] = deltas.get(r["dataset"], {}).get(r["id"], float("nan"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "dataset", "seed_ink_fraction", "gt_ink_fraction", "seed_to_gt_ratio",
            "stroke_width", "delta_f_stroke_minus_rel"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- 1. seed-mask ink fraction ----
    print("\n=== 1. Otsu seed mask vs ground truth: fraction of pixels marked as ink ===")
    print(f"{'dataset':<10} {'n':>3} {'GT ink %':>10} {'seed ink %':>11} "
          f"{'ratio median':>13} {'ratio min':>10} {'ratio max':>10} {'stroke px (med)':>16}")
    for ds in sorted({r["dataset"] for r in rows}):
        sel = [r for r in rows if r["dataset"] == ds]
        gt = np.array([r["gt_ink_fraction"] for r in sel]) * 100
        sd = np.array([r["seed_ink_fraction"] for r in sel]) * 100
        ratio = np.array([r["seed_to_gt_ratio"] for r in sel])
        sw = np.array([r["stroke_width"] for r in sel])
        print(f"{ds:<10} {len(sel):>3} {np.median(gt):>9.2f}% {np.median(sd):>10.2f}% "
              f"{np.nanmedian(ratio):>13.2f} {np.nanmin(ratio):>10.2f} {np.nanmax(ratio):>10.2f} "
              f"{np.median(sw):>16.2f}")

    # does a bad seed predict a worse stroke-width result? (pooled, rank correlation)
    pairs = [(np.log(r["seed_to_gt_ratio"]), r["delta_f_stroke_minus_rel"]) for r in rows
             if np.isfinite(r["seed_to_gt_ratio"]) and r["seed_to_gt_ratio"] > 0
             and np.isfinite(r["delta_f_stroke_minus_rel"])]
    if len(pairs) >= 10:
        rho, p = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        print(f"\nSpearman correlation, log(seed/GT ratio) vs delta F (all {len(pairs)} images): "
              f"rho={rho:+.3f}, p={p:.4f}")
        print("  (negative rho = the more the seed over-marks ink, the worse the stroke-width rule does)")

    # ---- 2. Wilcoxon ----
    print("\n=== 2. sauvola_stroke vs sauvola_rel: paired Wilcoxon signed-rank (two-sided) ===")
    print(f"{'dataset':<10} {'n':>3} {'mean d':>9} {'median d':>10} {'wins':>5} {'losses':>7} {'p-value':>9}")
    for ds in sorted(deltas):
        res = paired_wilcoxon(np.array(list(deltas[ds].values())))
        p_txt = f"{res['p']:.4f}" if np.isfinite(res["p"]) else "n/a"
        print(f"{ds:<10} {res['n']:>3} {res['mean']:>+9.4f} {res['median']:>+10.4f} "
              f"{res['wins']:>5} {res['losses']:>7} {p_txt:>9}")
    print("\n  p < 0.05 is the usual bar for calling a difference real. With only 10-20 images,")
    print("  a large p-value means 'no evidence either way', not 'proved identical'.")
    print(f"\nwrote per-image table -> {out}")


if __name__ == "__main__":
    main()
