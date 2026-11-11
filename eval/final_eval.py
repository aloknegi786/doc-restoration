"""Final evaluation: run the frozen pipeline on the two test sets.

STEP 8 — run this ONLY ONCE, after the pipeline is fully frozen.
Never adjust any parameter after seeing these numbers.

Test sets evaluated:
  DIBCO 2017    n=20
  Palm leaf     n=45  (GT1 and GT2 averaged per image)

The pipeline config is loaded from configs/pipeline_config.json if present,
otherwise from the frozen Sauvola + scale configs (configs/sauvola_config.json).

Usage (from project root, in .venv):
    python -m eval.final_eval \\
        --dataset dibco2017=data/dibco2017 \\
        --dataset palm=data/palm_split \\
        --splits data/splits.json \\
        --pipeline-config configs/pipeline_config.json \\
        --out eval/results/final_eval.csv

If --pipeline-config is omitted, the defaults in PipelineConfig are used
together with the frozen Sauvola params.  To specify the best channel found
in the ablation, pass e.g. --channel lab_l.
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

from docbin.config import PipelineConfig
from docbin.io import discover, load_sample
from docbin.pipeline import binarize
from eval.metrics import compute_metrics

METRIC_KEYS = ["precision", "recall", "f_measure", "dice", "psnr"]
METHOD_ORDER = ["full_pipeline", "human_gt1_vs_gt2"]


def mean_over_gts(pred: np.ndarray, gts: list[np.ndarray]) -> dict:
    ms = [compute_metrics(pred, g) for g in gts]
    return {k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS}


def load_pipeline_config(args) -> PipelineConfig:
    """Load the pipeline config, with channel override if requested."""
    if args.pipeline_config and Path(args.pipeline_config).exists():
        cfg = PipelineConfig.from_json(args.pipeline_config)
        print(f"Loaded pipeline config from {args.pipeline_config}")
    else:
        # Build from frozen Sauvola config
        sauvola_p = Path(args.sauvola_config)
        if not sauvola_p.exists():
            raise FileNotFoundError(
                f"Frozen Sauvola config not found: {sauvola_p}\n"
                "Run eval/param_sweep.py first, or pass --pipeline-config."
            )
        sauvola_cfg = json.loads(sauvola_p.read_text())
        cfg = PipelineConfig(
            window_rel=sauvola_cfg["window_rel"],
            k=sauvola_cfg["k"],
        )
        print(f"Built config from frozen Sauvola params: "
              f"window_rel={cfg.window_rel}, k={cfg.k}")

    if args.channel:
        import dataclasses
        cfg = dataclasses.replace(cfg, channel=args.channel)
        print(f"Channel override: {cfg.channel}")

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STEP 8: Final one-shot evaluation on DIBCO 2017 and palm test sets. "
            "Run once only after the pipeline is fully frozen."
        )
    )
    ap.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=DIR",
    )
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument(
        "--pipeline-config", default=None,
        help="Path to a pipeline_config.json. If absent, falls back to frozen Sauvola params.",
    )
    ap.add_argument(
        "--sauvola-config", default="configs/sauvola_config.json",
        help="Frozen (window_rel, k) — used only if --pipeline-config is absent.",
    )
    ap.add_argument(
        "--channel", default=None,
        help="Override the channel in the loaded config (e.g. lab_l).",
    )
    ap.add_argument("--out", default="eval/results/final_eval.csv")
    args = ap.parse_args()

    # -- Fail fast if splits are missing ------------------------------------
    splits_p = Path(args.splits)
    if not splits_p.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_p}")
    splits = json.loads(splits_p.read_text())

    # Test-set ids + exclude palm_dev
    test_ids: set[str] = set()
    for year_ids in splits.get("test", {}).values():
        test_ids.update(year_ids)
    palm_dev = set(splits.get("palm_dev", []))
    test_ids -= palm_dev

    cfg = load_pipeline_config(args)
    print(f"\nFull config:\n{cfg}\n")
    print(f"Test images: {len(test_ids)}")
    print("=" * 60)
    print("WARNING: these are the final test-set numbers. Do not")
    print("retune any parameter after viewing them.")
    print("=" * 60)

    rows: list[dict] = []
    for spec in args.dataset:
        ds_name, _, ds_path = spec.partition("=")
        samples = discover(ds_path, ds_name)
        n_scored = 0
        for s in samples:
            if s.id not in test_ids:
                continue
            bgr, gts = load_sample(s)
            t0 = time.perf_counter()
            mask = binarize(bgr, cfg)
            dt = time.perf_counter() - t0
            m = mean_over_gts(mask, gts)
            rows.append({
                "method": "full_pipeline",
                "id": s.id,
                "dataset": ds_name,
                "seconds": round(dt, 3),
                **m,
            })
            n_scored += 1
            # GT1 vs GT2 agreement for palm
            if len(gts) == 2:
                rows.append({
                    "method": "human_gt1_vs_gt2",
                    "id": s.id,
                    "dataset": ds_name,
                    "seconds": 0.0,
                    **compute_metrics(gts[0], gts[1]),
                })
        print(f"[{ds_name}] {n_scored} images scored")

    # -- Write CSV ----------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "id", "dataset", "seconds"] + METRIC_KEYS
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {out}")

    # -- Summary ------------------------------------------------------------
    print(f"\n{'dataset':<12} {'method':<20} {'n':>3} "
          f"{'F-measure':>16} {'PSNR (dB)':>12} {'Dice':>8}")
    print("-" * 75)
    for ds in sorted({r["dataset"] for r in rows}):
        for method in METHOD_ORDER:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == method]
            if not sel:
                continue
            f = np.array([r["f_measure"] for r in sel])
            p_all = np.array([r["psnr"] for r in sel])
            p_fin = p_all[np.isfinite(p_all)]
            d = np.array([r["dice"] for r in sel])
            psnr_txt = f"{p_fin.mean():8.2f}±{p_fin.std():.2f}" if len(p_fin) else "     inf"
            print(f"{ds:<12} {method:<20} {len(sel):>3} "
                  f"{f.mean():>8.4f}±{f.std():.4f} {psnr_txt:>12} {d.mean():>8.4f}")
    print(
        "\nReference: palm GT1-vs-GT2 F≈0.628 (inter-annotator agreement)."
        "\nAchievable range ≈ 0.63 – 0.81."
    )


if __name__ == "__main__":
    main()

