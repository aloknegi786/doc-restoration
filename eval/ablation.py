"""Ablation study: how much does each pipeline stage contribute?

Two sweeps are run, both on the **tuning sets only** (DIBCO 2018 + 2019,
n=10 each).  Test sets (DIBCO 2017, palm) are never touched here.

── Sweep 1: stage ablation ──────────────────────────────────────────────────
Each row adds one stage on top of the previous, starting from the best
tuned plain Sauvola (the strong baseline from run_baseline.py):

  sauvola_base  : channel=gray, Sauvola only (window_rel, k from frozen config)
  +cleanup      : + small-component removal
  +roi          : + ROI detection
  +normalise    : + masked background normalisation
  +clahe        : + CLAHE  (full pipeline, gray channel)

── Sweep 2: channel ablation ────────────────────────────────────────────────
Full pipeline (+clahe) with each of the four channels:

  channel=gray  (baseline, same as final row of sweep 1)
  channel=red
  channel=lab_l
  channel=pca

── Output ───────────────────────────────────────────────────────────────────
  --out      one CSV row per (image, config_name)
  terminal   per-config, per-dataset summary with paired Wilcoxon vs
             the immediately preceding row in each sweep

Usage (from project root, in .venv):
    python -m eval.ablation \\
        --dataset dibco2018=data/dibco2018 \\
        --dataset dibco2019=data/dibco2019 \\
        --splits data/splits.json \\
        --out eval/results/ablation.csv

Note: pass --also-test-sets to additionally score DIBCO 2017 and palm on
the *winning* full pipeline config — but only do this once, at the very end
(Step 8 of the project plan).  By default only tuning sets are scored.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.config import PipelineConfig
from docbin.io import discover, load_sample
from docbin.pipeline import binarize
from eval.metrics import compute_metrics

METRIC_KEYS = ["precision", "recall", "f_measure", "dice", "psnr"]


# ---------------------------------------------------------------------------
# Configuration sets
# ---------------------------------------------------------------------------

def build_stage_configs(base_cfg: PipelineConfig) -> "OrderedDict[str, PipelineConfig]":
    """Sequential stage ablation, starting from plain Sauvola.

    Stage order rationale:
      sauvola_base  : channel=gray, Sauvola only (frozen window_rel + k)
      +cleanup      : add small-component removal
      +normalise    : add background normalisation (full-frame, no ROI yet),
                      so its raw benefit on DIBCO is visible
      +roi          : add ROI-constrained normalisation (now the ROI mask
                      restricts the background estimate to the document region;
                      ROI without normalise has no effect on output so it is
                      not a useful separate step)
      +clahe        : add CLAHE on the normalised image
    """
    b = dataclasses.replace(
        base_cfg, channel="gray",
        use_roi=False, use_normalise=False, use_clahe=False, use_cleanup=False,
    )
    return OrderedDict([
        ("sauvola_base", b),
        ("+cleanup",     dataclasses.replace(b, use_cleanup=True)),
        ("+normalise",   dataclasses.replace(b, use_cleanup=True,
                                              use_normalise=True)),
        ("+roi",         dataclasses.replace(b, use_cleanup=True,
                                              use_normalise=True, use_roi=True)),
        ("+clahe",       dataclasses.replace(b, use_cleanup=True,
                                              use_normalise=True, use_roi=True,
                                              use_clahe=True)),
    ])


def build_channel_configs(full_cfg: PipelineConfig) -> "OrderedDict[str, PipelineConfig]":
    """Channel ablation: full pipeline with each of the four channels."""
    return OrderedDict([
        (f"channel={ch}", dataclasses.replace(full_cfg, channel=ch))
        for ch in ("gray", "red", "lab_l", "pca")
    ])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def mean_over_gts(pred: np.ndarray, gts: list[np.ndarray]) -> dict:
    ms = [compute_metrics(pred, g) for g in gts]
    return {k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}


def score_configs(
    datasets: list[tuple[str, str]],
    configs: "OrderedDict[str, PipelineConfig]",
    tuning_ids: set[str],
    also_test_ids: set[str] | None = None,
) -> list[dict]:
    """Run every config on every eligible image and return rows.

    Parameters
    ----------
    datasets   : list of (name, root_path) pairs.
    configs    : ordered dict of config_name → PipelineConfig.
    tuning_ids : set of sample ids that belong to the tuning sets.
    also_test_ids : if provided, also score these ids (use only for Step 8).
    """
    eligible = tuning_ids | (also_test_ids or set())
    rows: list[dict] = []

    for cfg_name, cfg in configs.items():
        print(f"\n  [{cfg_name}]")
        for ds_name, ds_path in datasets:
            samples = discover(ds_path, ds_name)
            n_scored = 0
            for s in samples:
                if s.id not in eligible:
                    continue
                bgr, gts = load_sample(s)
                t0 = time.perf_counter()
                mask = binarize(bgr, cfg)
                dt = time.perf_counter() - t0
                m = mean_over_gts(mask, gts)
                rows.append({
                    "config": cfg_name,
                    "id": s.id,
                    "dataset": ds_name,
                    "seconds": round(dt, 3),
                    **m,
                })
                n_scored += 1
            print(f"    {ds_name}: {n_scored} images scored")
    return rows


# ---------------------------------------------------------------------------
# Wilcoxon helper
# ---------------------------------------------------------------------------

def paired_wilcoxon(deltas: np.ndarray) -> tuple[float, float]:
    """Return (mean_delta, p_value).  p=nan if too few non-zero pairs."""
    deltas = np.asarray(deltas, dtype=float)
    if np.count_nonzero(deltas) < 6:
        return float(deltas.mean()), float("nan")
    p = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue
    return float(deltas.mean()), float(p)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _get_f(rows: list[dict], config: str, dataset: str) -> np.ndarray:
    return np.array([r["f_measure"] for r in rows
                     if r["config"] == config and r["dataset"] == dataset])


def _get_psnr(rows: list[dict], config: str, dataset: str) -> np.ndarray:
    return np.array([r["psnr"] for r in rows
                     if r["config"] == config and r["dataset"] == dataset])


def print_sweep_summary(
    rows: list[dict],
    configs: "OrderedDict[str, PipelineConfig]",
    datasets: list[str],
    title: str,
) -> None:
    cfg_names = list(configs.keys())
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    hdr = f"{'config':<18}"
    for ds in datasets:
        hdr += f"  {'F-measure (' + ds + ')':<22}"
    hdr += f"  {'Wilcoxon p (per dataset)'}"
    print(hdr)
    print("-" * 80)

    for idx, cfg_name in enumerate(cfg_names):
        line = f"{cfg_name:<18}"
        p_parts = []
        for ds in datasets:
            f = _get_f(rows, cfg_name, ds)
            if len(f) == 0:
                line += f"  {'—':^22}"
                continue
            line += f"  {f.mean():.4f} ± {f.std():.4f}       "

            # Paired delta vs previous config on same dataset
            if idx == 0:
                p_parts.append("—")
            else:
                prev = cfg_names[idx - 1]
                prev_f = _get_f(rows, prev, ds)
                # Match by sample id
                prev_map = {r["id"]: r["f_measure"] for r in rows
                            if r["config"] == prev and r["dataset"] == ds}
                curr_map = {r["id"]: r["f_measure"] for r in rows
                            if r["config"] == cfg_name and r["dataset"] == ds}
                common = sorted(set(prev_map) & set(curr_map))
                if common:
                    deltas = np.array([curr_map[i] - prev_map[i] for i in common])
                    mean_d, p = paired_wilcoxon(deltas)
                    wins = int((deltas > 0).sum())
                    p_txt = f"{p:.3f}" if np.isfinite(p) else "n/a"
                    sig = " ✓" if np.isfinite(p) and p < 0.05 else ""
                    p_parts.append(f"{ds}: Δ{mean_d:+.4f} wins={wins}/{len(common)} p={p_txt}{sig}")
                else:
                    p_parts.append(f"{ds}: —")

        line += "  " + " | ".join(p_parts)
        print(line)

    # Also print PSNR for completeness
    print(f"\n{'config':<18}", end="")
    for ds in datasets:
        print(f"  {'PSNR (' + ds + ') dB':<22}", end="")
    print()
    print("-" * 60)
    for cfg_name in cfg_names:
        print(f"{cfg_name:<18}", end="")
        for ds in datasets:
            p = _get_psnr(rows, cfg_name, ds)
            p_fin = p[np.isfinite(p)]
            if len(p_fin) == 0:
                print(f"  {'—':^22}", end="")
            else:
                print(f"  {p_fin.mean():.2f} ± {p_fin.std():.2f}            ", end="")
        print()


def print_best_channel(
    rows: list[dict],
    channel_configs: "OrderedDict[str, PipelineConfig]",
    datasets: list[str],
) -> str:
    """Print channel comparison and return the best channel name."""
    print(f"\n{'='*80}")
    print("Channel ablation (full pipeline)")
    print(f"{'='*80}")
    print(f"{'config':<20}", end="")
    for ds in datasets:
        print(f"  F ± std ({ds})         ", end="")
    print(f"  F pooled")
    print("-" * 80)

    best_ch = "gray"
    best_pooled = -1.0
    for cfg_name in channel_configs:
        print(f"{cfg_name:<20}", end="")
        pooled = []
        for ds in datasets:
            f = _get_f(rows, cfg_name, ds)
            if len(f) == 0:
                print(f"  {'—':^22}", end="")
            else:
                print(f"  {f.mean():.4f} ± {f.std():.4f}       ", end="")
                pooled.extend(f.tolist())
        mean_pooled = float(np.mean(pooled)) if pooled else -1.0
        marker = "  ← best" if mean_pooled > best_pooled else ""
        if mean_pooled > best_pooled:
            best_pooled = mean_pooled
            best_ch = cfg_name.split("=")[1]
        print(f"  {mean_pooled:.4f}{marker}")
    print(f"\nBest channel on tuning set: {best_ch}  (pooled F = {best_pooled:.4f})")
    print("NOTE: pick channel for the final pipeline config based on this, then")
    print("run eval/final_eval.py ONCE on DIBCO 2017 + palm test sets.")
    return best_ch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the ablation study on the tuning sets."
    )
    ap.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=DIR",
        help="Dataset name and path (use tuning datasets: dibco2018, dibco2019).",
    )
    ap.add_argument(
        "--splits", default="data/splits.json",
        help="Path to splits.json.",
    )
    ap.add_argument(
        "--sauvola-config", default="configs/sauvola_config.json",
        help="Frozen (window_rel, k) from param_sweep.py.",
    )
    ap.add_argument(
        "--out", default="eval/results/ablation.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--also-test-sets", action="store_true",
        help=(
            "STEP 8 ONLY: also score test-set ids found in --dataset paths. "
            "Do NOT use until the pipeline is fully frozen."
        ),
    )
    args = ap.parse_args()

    # -- Validate config files are present ----------------------------------
    sauvola_p = Path(args.sauvola_config)
    if not sauvola_p.exists():
        raise FileNotFoundError(
            f"Frozen Sauvola config not found: {sauvola_p}\n"
            "Run eval/param_sweep.py first."
        )
    sauvola_cfg = json.loads(sauvola_p.read_text())
    base_cfg = PipelineConfig(
        window_rel=sauvola_cfg["window_rel"],
        k=sauvola_cfg["k"],
    )
    print(f"Loaded frozen Sauvola params: window_rel={base_cfg.window_rel}, k={base_cfg.k}")

    # -- Load splits --------------------------------------------------------
    splits_p = Path(args.splits)
    if not splits_p.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_p}")
    splits = json.loads(splits_p.read_text())

    tuning_ids: set[str] = set()
    for year_ids in splits.get("tuning", {}).values():
        tuning_ids.update(year_ids)

    test_ids: set[str] = set()
    if args.also_test_sets:
        for year_ids in splits.get("test", {}).values():
            test_ids.update(year_ids)
        # Exclude palm_dev
        test_ids -= set(splits.get("palm_dev", []))
        print(f"[--also-test-sets] will also score {len(test_ids)} test images")

    palm_dev = set(splits.get("palm_dev", []))
    eligible_all = tuning_ids | test_ids

    # -- Parse dataset specs ------------------------------------------------
    datasets_named: list[tuple[str, str]] = []
    for spec in args.dataset:
        name, _, path = spec.partition("=")
        datasets_named.append((name, path))

    tuning_ds_names = sorted({
        name for name, _ in datasets_named
        if any(sid.startswith(name + "/") for sid in tuning_ids)
    })

    # -- Build configs ------------------------------------------------------
    stage_configs = build_stage_configs(base_cfg)
    # Base for channel ablation: the best-performing stage combination
    # (+cleanup +normalise, no ROI, no CLAHE), as determined by tuning evidence.
    # Using the degraded "+clahe" config as base would give unreliable channel rankings.
    channel_base_cfg = stage_configs["+normalise"]
    channel_configs = build_channel_configs(channel_base_cfg)

    all_configs: "OrderedDict[str, PipelineConfig]" = OrderedDict()
    all_configs.update(stage_configs)
    # channel=gray is already the same as +clahe; add red/lab_l/pca only
    for k, v in channel_configs.items():
        if k != "channel=gray":
            all_configs[k] = v

    print(f"\nRunning {len(all_configs)} configs × tuning images "
          f"(+ {len(test_ids)} test images if --also-test-sets) ...")

    rows = score_configs(datasets_named, all_configs, tuning_ids, test_ids or None)

    # Re-alias +normalise (which is gray) as channel=gray for the channel table
    for r in rows:
        if r["config"] == "+normalise":
            rows.append({**r, "config": "channel=gray"})

    # -- Write CSV ----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["config", "id", "dataset", "seconds"] + METRIC_KEYS
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {out_path}")

    # -- Print summaries ----------------------------------------------------
    print_sweep_summary(
        rows, stage_configs, tuning_ds_names,
        "Stage ablation  (tuning sets only — do NOT adjust parameters after seeing these)"
    )
    best_ch = print_best_channel(rows, channel_configs, tuning_ds_names)

    if args.also_test_sets:
        test_ds_names = sorted({
            name for name, _ in datasets_named
            if any(sid.startswith(name + "/") for sid in test_ids)
        })
        if test_ds_names:
            print(f"\n{'='*80}")
            print("TEST SET RESULTS (Step 8 — final, do not retune after this)")
            print(f"{'='*80}")
            # Show full pipeline (channel=best) on test sets
            best_cfg_name = f"channel={best_ch}" if best_ch != "gray" else "+clahe"
            for ds in test_ds_names:
                f = _get_f(rows, best_cfg_name, ds)
                p = _get_psnr(rows, best_cfg_name, ds)
                p_fin = p[np.isfinite(p)]
                if len(f) > 0:
                    print(f"  {ds:15}  n={len(f):3}  "
                          f"F={f.mean():.4f}±{f.std():.4f}  "
                          f"PSNR={p_fin.mean():.2f}±{p_fin.std():.2f} dB")

    print(
        f"\nNext: inspect the stage Wilcoxon p-values above. "
        f"Then run eval/final_eval.py on DIBCO 2017 + palm (once, frozen pipeline)."
    )


if __name__ == "__main__":
    main()

