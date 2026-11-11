"""Baseline experiment: grayscale -> Otsu, and grayscale -> Sauvola.

Two Sauvola variants are run side by side, so their gap is visible in every
results table from here on:
  * sauvola_rel    : window_from_relative (old placeholder rule)
  * sauvola_stroke : estimate_window, the stroke-width based rule (current)

k is shared between both variants -- it comes from the param sweep, which
was tuned for the relative-window rule. Only the window size differs.

By default the frozen configs are read from eval/results/sauvola_config.json
(window_rel, k) and eval/results/scale_config.json (factor, floor), i.e. the
outputs of param_sweep.py and calibrate_scale.py. Pass --window-rel / --k /
--factor / --floor to override any of them for a one-off experiment; doing
so does NOT update the frozen config files.

Usage:
    python -m eval.run_baseline \
        --dataset dibco2018=data/dibco2018 \
        --dataset dibco2019=data/dibco2019 \
        --dataset dibco2017=data/dibco2017 \
        --dataset palm=data/palm_split \
        --splits data/splits.json \
        --out eval/results/baseline_v2.csv

IMPORTANT: sauvola_rel/sauvola_stroke's parameters were tuned/calibrated on
DIBCO 2018+2019 only. Do not adjust them by looking at the DIBCO 2017 or
palm leaf numbers produced here; those are test sets.

Outputs one row per (image, method). Palm leaf images have two GTs; their
metrics are the mean over both. A 'human_gt1_vs_gt2' pseudo-method records
agreement between the two annotators (the human ceiling).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover, load_sample, to_gray
from docbin.scale import DEFAULT_FACTOR, DEFAULT_FLOOR, estimate_window
from docbin.segmentation import otsu_binarize, sauvola_binarize, window_from_relative
from eval.metrics import compute_metrics

METRIC_KEYS = ["precision", "recall", "f_measure", "dice", "psnr"]
METHOD_ORDER = ["otsu", "sauvola_rel", "sauvola_stroke", "human_gt1_vs_gt2"]


def mean_over_gts(pred, gts):
    ms = [compute_metrics(pred, g) for g in gts]
    return {k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}


def load_frozen_params(args) -> tuple[float, float, float, int]:
    """Resolve (window_rel, k, factor, floor), CLI overrides winning over the
    frozen config files, which win over the module defaults.

    Raises FileNotFoundError if a config path was supplied but does not exist
    AND no CLI override was given for its parameters.  Silent fallback to
    un-tuned defaults previously produced invalid results tables; we now fail
    loudly instead.
    """
    window_rel, k, factor, floor = None, None, DEFAULT_FACTOR, DEFAULT_FLOOR

    # --- sauvola config ---
    if args.sauvola_config:
        p = Path(args.sauvola_config)
        if not p.exists():
            # Only hard-fail when the caller did not supply both overrides.
            if args.window_rel is None or args.k is None:
                raise FileNotFoundError(
                    f"sauvola config not found: {p}\n"
                    "Run eval/param_sweep.py first, or pass --window-rel and --k."
                )
        else:
            cfg = json.loads(p.read_text())
            window_rel, k = cfg["window_rel"], cfg["k"]

    # --- scale config ---
    if args.scale_config:
        p = Path(args.scale_config)
        if not p.exists():
            if args.factor is None:
                raise FileNotFoundError(
                    f"scale config not found: {p}\n"
                    "Run eval/calibrate_scale.py first, or pass --factor."
                )
        else:
            cfg = json.loads(p.read_text())
            factor, floor = cfg["factor"], cfg["floor"]

    # CLI overrides always win; apply after reading files.
    if args.window_rel is not None:
        window_rel = args.window_rel
    if args.k is not None:
        k = args.k
    if args.factor is not None:
        factor = args.factor
    if args.floor is not None:
        floor = args.floor

    # Final safety check — both must be resolved.
    if window_rel is None or k is None:
        raise RuntimeError(
            "window_rel / k were not resolved. "
            "Supply --sauvola-config or --window-rel + --k."
        )

    return window_rel, k, factor, floor


def run(args, window_rel: float, k: float, factor: float, floor: int) -> list[dict]:
    excluded: set[str] = set()
    if args.splits:
        excluded = set(json.loads(Path(args.splits).read_text()).get("palm_dev", []))

    rows: list[dict] = []
    for spec in args.dataset:
        name, _, path = spec.partition("=")
        samples = discover(path, name)
        n_skip = 0
        for s in samples:
            if s.id in excluded:
                n_skip += 1
                continue
            bgr, gts = load_sample(s)
            gray = to_gray(bgr)
            h, w = gray.shape
            base = {"id": s.id, "dataset": name, "height": h, "width": w}

            t0 = time.perf_counter()
            pred = otsu_binarize(gray)
            dt = time.perf_counter() - t0
            rows.append({**base, "method": "otsu", "window_px": None, "seconds": dt,
                         **mean_over_gts(pred, gts)})

            win_rel_px = window_from_relative(gray.shape, window_rel)
            t0 = time.perf_counter()
            pred = sauvola_binarize(gray, window=win_rel_px, k=k)
            dt = time.perf_counter() - t0
            rows.append({**base, "method": "sauvola_rel", "window_px": win_rel_px, "seconds": dt,
                         **mean_over_gts(pred, gts)})

            win_stroke_px = estimate_window(gray, factor=factor, floor=floor)
            t0 = time.perf_counter()
            pred = sauvola_binarize(gray, window=win_stroke_px, k=k)
            dt = time.perf_counter() - t0
            rows.append({**base, "method": "sauvola_stroke", "window_px": win_stroke_px, "seconds": dt,
                         **mean_over_gts(pred, gts)})

            if len(gts) == 2:
                rows.append({**base, "method": "human_gt1_vs_gt2", "window_px": None, "seconds": 0.0,
                             **compute_metrics(gts[0], gts[1])})
        print(f"[{name}] {len(samples) - n_skip} images scored"
              + (f" ({n_skip} dev images excluded)" if n_skip else ""))
    return rows


def summarise(rows: list[dict]) -> None:
    print(f"\n{'dataset':<10} {'method':<16} {'n':>3} {'F-measure':>16} {'PSNR':>14} {'Dice':>8}")
    for ds in sorted({r['dataset'] for r in rows}):
        for m in METHOD_ORDER:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            if not sel:
                continue
            f = np.array([r["f_measure"] for r in sel])
            p_all = np.array([r["psnr"] for r in sel])
            p = p_all[np.isfinite(p_all)]
            d = np.array([r["dice"] for r in sel])
            psnr_txt = f"{p.mean():>8.2f}±{p.std():.2f}" if len(p) else f"{'inf':>8}"
            if 0 < len(p) < len(p_all):
                psnr_txt += "*"          # * = some images were pixel-perfect (inf) and excluded
            print(f"{ds:<10} {m:<16} {len(sel):>3} {f.mean():>8.4f}±{f.std():.4f} "
                  f"{psnr_txt:>14} {d.mean():>8.4f}")

    # paired sauvola_rel vs sauvola_stroke delta, per dataset
    print(f"\n{'dataset':<10} {'sauvola_stroke - sauvola_rel (paired, per image)':<50}")
    for ds in sorted({r['dataset'] for r in rows}):
        rel = {r["id"]: r["f_measure"] for r in rows if r["dataset"] == ds and r["method"] == "sauvola_rel"}
        stroke = {r["id"]: r["f_measure"] for r in rows if r["dataset"] == ds and r["method"] == "sauvola_stroke"}
        common = sorted(set(rel) & set(stroke))
        if not common:
            continue
        deltas = np.array([stroke[i] - rel[i] for i in common])
        wins = int((deltas > 0).sum())
        print(f"{ds:<10} mean delta={deltas.mean():+.4f}  wins {wins}/{len(common)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--splits", help="splits.json (palm_dev images are excluded)")
    ap.add_argument("--sauvola-config", default="eval/results/sauvola_config.json",
                     help="frozen (window_rel, k) from param_sweep.py; empty string to skip")
    ap.add_argument("--scale-config", default="eval/results/scale_config.json",
                     help="frozen (factor, floor) from calibrate_scale.py; empty string to skip")
    ap.add_argument("--window-rel", type=float, default=None, help="override window_rel")
    ap.add_argument("--k", type=float, default=None, help="override k")
    ap.add_argument("--factor", type=float, default=None, help="override stroke-width factor")
    ap.add_argument("--floor", type=int, default=None, help="override window floor (px)")
    ap.add_argument("--out", default="eval/results/baseline_v2.csv")
    args = ap.parse_args()

    window_rel, k, factor, floor = load_frozen_params(args)
    print(f"using: window_rel={window_rel}, k={k}, factor={factor}, floor={floor}\n")

    rows = run(args, window_rel, k, factor, floor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "dataset", "method", "window_px", "height", "width", "seconds"] + METRIC_KEYS
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    summarise(rows)


if __name__ == "__main__":
    main()
