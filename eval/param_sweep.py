"""Parameter sweep for Sauvola: window size (relative) and k.

Runs ONLY on the tuning sets (DIBCO 2018 + 2019), using the cross-year folds
from splits.json:
    fold A: grid-search on 2018, validate the chosen point on 2019
    fold B: grid-search on 2019, validate the chosen point on 2018

For each (window_rel, k) grid point we compute the MEAN F-Measure over the
training year's images, pick the best point on that year alone, then report
how that same point does on the other year (validation). This is what makes
the result a fair estimate of what to expect on unseen data, rather than a
number optimised directly against it.

STAGE 2 -- refinement. The two folds' coarse optima are rarely identical, and
picking their coordinate-wise average is a point neither fold actually
evaluated. Instead we build a small, finer grid spanning the two coarse
optima (with a margin), score BOTH years at every refined point, and pick
the point that maximises the WORST-CASE (minimum) F-Measure across the two
years. This protects against a config that is excellent on one year's style
of degradation and poor on the other's, which is the failure mode fold B
showed in the coarse sweep (0.755 train -> 0.615 validate).

Never touches dibco2017 or palm. Their turn comes only in eval/run_eval.py,
once, with the frozen config this script produces.

Usage:
    python -m eval.param_sweep --dibco2018 data/dibco2018 --dibco2019 data/dibco2019 \
        --splits data/splits.json --out eval/results/param_sweep.csv \
        --config-out eval/results/sauvola_config.json
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
from docbin.segmentation import sauvola_binarize, window_from_relative
from eval.metrics import compute_metrics

# Coarse grid, chosen deliberately small for a 10-image tuning year (see the
# project notes on overfitting with few images). Widen it only if the chosen
# point sits at a grid edge (see the warning printed at the end).
WINDOW_RELS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12]
KS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# Refinement step sizes (stage 2). The refined grid spans the two coarse
# optima plus one step of margin on each side, at this resolution.
REFINE_STEP_REL = 0.01
REFINE_STEP_K = 0.02


def load_year(root: str | Path, dataset: str, ids: list[str]) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    """Load grayscale image + GT masks for exactly the ids listed in splits.json."""
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


def mean_f_measure(pred: np.ndarray, gts: list[np.ndarray]) -> float:
    return float(np.mean([compute_metrics(pred, g)["f_measure"] for g in gts]))


def score_grid(images: list[tuple[str, np.ndarray, list[np.ndarray]]]) -> dict:
    """Return {(window_rel, k): {"mean_f": ..., "per_image": {id: f}}}."""
    grid: dict[tuple[float, float], dict] = {}
    for rel in WINDOW_RELS:
        for k in KS:
            per_image = {}
            for img_id, gray, gts in images:
                win = window_from_relative(gray.shape, rel)
                pred = sauvola_binarize(gray, window=win, k=k)
                per_image[img_id] = mean_f_measure(pred, gts)
            fs = np.array(list(per_image.values()))
            grid[(rel, k)] = {"mean_f": float(fs.mean()), "std_f": float(fs.std()), "per_image": per_image}
    return grid


def best_point(grid: dict) -> tuple[float, float]:
    return max(grid, key=lambda pt: grid[pt]["mean_f"])


def edge_warning(rel: float, k: float) -> str | None:
    msgs = []
    if rel in (WINDOW_RELS[0], WINDOW_RELS[-1]):
        msgs.append(f"window_rel={rel} is at the grid edge")
    if k in (KS[0], KS[-1]):
        msgs.append(f"k={k} is at the grid edge")
    return "; ".join(msgs) if msgs else None


def run_fold(name: str, train, validate) -> dict:
    train_grid = score_grid(train)
    rel, k = best_point(train_grid)
    val_grid = score_grid(validate)                 # full grid on validate, so we can also inspect it
    val_at_point = val_grid[(rel, k)]

    print(f"\n=== fold {name}: tune on {len(train)} images, validate on {len(validate)} ===")
    print(f"  chosen point: window_rel={rel}, k={k}  "
          f"(train mean F={train_grid[(rel, k)]['mean_f']:.4f}, "
          f"validate mean F={val_at_point['mean_f']:.4f})")
    warn = edge_warning(rel, k)
    if warn:
        print(f"  [warn] {warn} -- consider widening the grid")

    return {
        "fold": name,
        "window_rel": rel,
        "k": k,
        "train_mean_f": train_grid[(rel, k)]["mean_f"],
        "validate_mean_f": val_at_point["mean_f"],
        "train_grid": train_grid,
        "validate_grid": val_grid,
    }


def frange(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive float range, rounded to avoid binary-fraction drift."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def refined_grid_points(fold_a: dict, fold_b: dict) -> tuple[list[float], list[float]]:
    """Span the two coarse optima with one step of margin on each side."""
    rels = [fold_a["window_rel"], fold_b["window_rel"]]
    ks = [fold_a["k"], fold_b["k"]]
    rel_lo = max(REFINE_STEP_REL, min(rels) - REFINE_STEP_REL)
    rel_hi = max(rels) + REFINE_STEP_REL
    k_lo = max(REFINE_STEP_K, min(ks) - REFINE_STEP_K)
    k_hi = max(ks) + REFINE_STEP_K
    return frange(rel_lo, rel_hi, REFINE_STEP_REL), frange(k_lo, k_hi, REFINE_STEP_K)


def score_grid_points(images, rels: list[float], ks: list[float]) -> dict:
    """Same as score_grid but over an explicit (rels, ks) list instead of the module globals."""
    grid: dict[tuple[float, float], dict] = {}
    for rel in rels:
        for k in ks:
            per_image = {}
            for img_id, gray, gts in images:
                win = window_from_relative(gray.shape, rel)
                pred = sauvola_binarize(gray, window=win, k=k)
                per_image[img_id] = mean_f_measure(pred, gts)
            fs = np.array(list(per_image.values()))
            grid[(rel, k)] = {"mean_f": float(fs.mean()), "std_f": float(fs.std()), "per_image": per_image}
    return grid


def refine(fold_a: dict, fold_b: dict, imgs18, imgs19) -> dict:
    """Stage 2: finer grid around the coarse optima, scored on BOTH years,
    final point = highest worst-case (min across the two years) F-Measure."""
    rels, ks = refined_grid_points(fold_a, fold_b)
    print(f"\n=== refinement: window_rel in {rels}, k in {ks} "
          f"({len(rels) * len(ks)} points, scored on both years) ===")

    grid_2018 = score_grid_points(imgs18, rels, ks)
    grid_2019 = score_grid_points(imgs19, rels, ks)

    scored = []
    for pt in grid_2018:
        f18, f19 = grid_2018[pt]["mean_f"], grid_2019[pt]["mean_f"]
        scored.append({"point": pt, "f_2018": f18, "f_2019": f19,
                        "mean": (f18 + f19) / 2, "min": min(f18, f19)})

    best_worst_case = max(scored, key=lambda r: r["min"])
    best_mean = max(scored, key=lambda r: r["mean"])

    print(f"  best worst-case point: window_rel={best_worst_case['point'][0]}, "
          f"k={best_worst_case['point'][1]}  "
          f"(F_2018={best_worst_case['f_2018']:.4f}, F_2019={best_worst_case['f_2019']:.4f}, "
          f"min={best_worst_case['min']:.4f})")
    if best_mean["point"] != best_worst_case["point"]:
        print(f"  [note] best-mean point differs: window_rel={best_mean['point'][0]}, "
              f"k={best_mean['point'][1]}  "
              f"(F_2018={best_mean['f_2018']:.4f}, F_2019={best_mean['f_2019']:.4f}, "
              f"mean={best_mean['mean']:.4f}) -- kept for the report, not selected")

    return {
        "rels": rels, "ks": ks,
        "grid_2018": grid_2018, "grid_2019": grid_2019,
        "scored": scored,
        "selected": best_worst_case,
        "best_mean_alternative": best_mean,
    }


def write_grid_csv(path: Path, folds: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "phase", "window_rel", "k", "mean_f", "std_f"])
        for fold in folds:
            for phase, grid in (("train", fold["train_grid"]), ("validate", fold["validate_grid"])):
                for (rel, k), res in grid.items():
                    w.writerow([fold["fold"], phase, rel, k, res["mean_f"], res["std_f"]])


def write_refine_csv(path: Path, refined: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_rel", "k", "f_2018", "f_2019", "mean", "min", "selected"])
        for row in refined["scored"]:
            sel = row["point"] == refined["selected"]["point"]
            w.writerow([row["point"][0], row["point"][1], row["f_2018"], row["f_2019"],
                        row["mean"], row["min"], sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dibco2018", required=True)
    ap.add_argument("--dibco2019", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="eval/results/param_sweep.csv")
    ap.add_argument("--refine-out", default="eval/results/param_sweep_refined.csv")
    ap.add_argument("--config-out", default="eval/results/sauvola_config.json")
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    ids18 = splits["tuning"]["dibco2018"]
    ids19 = splits["tuning"]["dibco2019"]
    if not ids18 or not ids19:
        raise SystemExit("error: splits.json has an empty tuning set for 2018 or 2019")

    imgs18 = load_year(args.dibco2018, "dibco2018", ids18)
    imgs19 = load_year(args.dibco2019, "dibco2019", ids19)

    fold_a = run_fold("A (tune=2018, validate=2019)", imgs18, imgs19)
    fold_b = run_fold("B (tune=2019, validate=2018)", imgs19, imgs18)

    write_grid_csv(Path(args.out), [fold_a, fold_b])

    if fold_a["window_rel"] != fold_b["window_rel"] or fold_a["k"] != fold_b["k"]:
        print("\n[note] the two folds picked different coarse optima -- refining "
              "around both instead of averaging them blindly.")

    # Stage 2: refine around the two coarse optima, scored on BOTH years,
    # selected by worst-case (minimum) F-Measure across the two years. This
    # is the point actually evaluated on both, unlike a coordinate average.
    refined = refine(fold_a, fold_b, imgs18, imgs19)
    write_refine_csv(Path(args.refine_out), refined)

    sel = refined["selected"]
    final_rel, final_k = sel["point"]

    config = {
        "method": "sauvola",
        "window_rel": final_rel,
        "k": final_k,
        "selection_rule": "worst_case_min_over_2018_2019",
        "note": "window_rel is a placeholder rule (fraction of shorter image "
                "side); replace with a stroke-width estimate before final report.",
        "f_2018": sel["f_2018"],
        "f_2019": sel["f_2019"],
        "worst_case_f": sel["min"],
        "mean_f": sel["mean"],
        "coarse_fold_a": {"window_rel": fold_a["window_rel"], "k": fold_a["k"], "validate_mean_f": fold_a["validate_mean_f"]},
        "coarse_fold_b": {"window_rel": fold_b["window_rel"], "k": fold_b["k"], "validate_mean_f": fold_b["validate_mean_f"]},
        "best_mean_alternative": {
            "window_rel": refined["best_mean_alternative"]["point"][0],
            "k": refined["best_mean_alternative"]["point"][1],
            "mean_f": refined["best_mean_alternative"]["mean"],
        },
    }
    Path(args.config_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_out).write_text(json.dumps(config, indent=2))

    print(f"\nfinal config (worst-case selection): window_rel={final_rel}, k={final_k}")
    print(f"  F_2018={sel['f_2018']:.4f}  F_2019={sel['f_2019']:.4f}  "
          f"worst-case={sel['min']:.4f}  mean={sel['mean']:.4f}")
    print(f"wrote coarse grid -> {args.out}")
    print(f"wrote refined grid -> {args.refine_out}")
    print(f"wrote config -> {args.config_out}")


if __name__ == "__main__":
    main()
