"""Failure analysis figures for the report.

Generates two types of figure:

1. Per-image comparison panels (worst images per dataset):
   original | GT | sauvola_base | best pipeline
   Saved as eval/results/figures/<dataset>_worst_<n>.png

2. Parameter sensitivity heatmap:
   F-measure on DIBCO 2018 and 2019 as a function of (window_rel, k)
   from eval/results/param_sweep_refined.csv
   Saved as eval/results/figures/param_sensitivity.png

Usage (from project root, in .venv):
    python -m eval.failure_analysis \\
        --dataset dibco2017=data/dibco2017 \\
        --dataset dibco2018=data/dibco2018 \\
        --dataset dibco2019=data/dibco2019 \\
        --dataset palm=data/palm_split \\
        --splits data/splits.json \\
        --results-csv eval/results/ablation.csv \\
        --n-worst 3 \\
        --channel lab_l \\
        --out-dir eval/results/figures

The --results-csv may be either ablation.csv or final_eval.csv.
If final_eval.csv exists, prefer it for test-set results.
If only ablation.csv is available, tuning-set worst images are shown.

Note on column names: ablation.csv has a "config" column; final_eval.csv
has a "method" column.  The script auto-detects which file is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.config import PipelineConfig
from docbin.io import discover, load_sample
from docbin.pipeline import run as pipeline_run
from docbin.segmentation import sauvola_binarize, window_from_relative


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_base_cfg(sauvola_config: str) -> PipelineConfig:
    p = Path(sauvola_config)
    if not p.exists():
        raise FileNotFoundError(f"Sauvola config not found: {p}")
    sc = json.loads(p.read_text())
    return PipelineConfig(window_rel=sc["window_rel"], k=sc["k"])


def _make_base_pipeline(base_cfg: PipelineConfig) -> PipelineConfig:
    """sauvola_base: Sauvola only, gray channel."""
    import dataclasses
    return dataclasses.replace(
        base_cfg, channel="gray",
        use_roi=False, use_normalise=False, use_clahe=False, use_cleanup=False,
    )


def _make_full_pipeline(base_cfg: PipelineConfig, channel: str) -> PipelineConfig:
    """Best pipeline: cleanup + normalise, chosen channel."""
    import dataclasses
    return dataclasses.replace(
        base_cfg, channel=channel,
        use_roi=False, use_normalise=True, use_clahe=False, use_cleanup=True,
    )


# ---------------------------------------------------------------------------
# Load worst images from a results CSV
# ---------------------------------------------------------------------------

def _load_worst(csv_path: str, n: int) -> dict[str, list[str]]:
    """Return {dataset: [worst image ids]} sorted ascending by F-measure."""
    rows: list[dict] = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    # Detect column name for 'config' or 'method'
    id_col = "config" if "config" in (rows[0] if rows else {}) else "method"

    # For ablation CSV: use 'sauvola_base' or first unique config.
    # For final_eval CSV: use 'full_pipeline'.
    if id_col == "config":
        target = "sauvola_base"
    else:
        target = "full_pipeline"

    best_rows = [r for r in rows if r.get(id_col) == target]
    if not best_rows:
        # Fallback: use all rows
        best_rows = rows

    worst: dict[str, list[str]] = {}
    for ds in sorted({r["dataset"] for r in best_rows}):
        ds_rows = sorted(
            [r for r in best_rows if r["dataset"] == ds],
            key=lambda r: float(r["f_measure"]),
        )
        worst[ds] = [r["id"] for r in ds_rows[:n]]
    return worst


# ---------------------------------------------------------------------------
# Mask visualisation
# ---------------------------------------------------------------------------

def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert bool ink mask → 3-channel BGR (ink=black, paper=white)."""
    gray = np.where(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _gt_mean(gts: list[np.ndarray]) -> np.ndarray:
    """Average multiple GT masks into a single display image."""
    if len(gts) == 1:
        return _mask_to_bgr(gts[0])
    # For palm (2 GTs): show pixels where both agree
    agree = gts[0] & gts[1]
    disagree = gts[0] ^ gts[1]
    vis = np.full((*gts[0].shape, 3), 255, dtype=np.uint8)
    vis[agree] = (0, 0, 0)          # both say ink → black
    vis[disagree] = (0, 150, 255)   # disagreement → orange
    return vis


def _add_label(img: np.ndarray, text: str, f: float | None = None) -> np.ndarray:
    """Overlay a small text label at the top of an image."""
    out = img.copy()
    label = text if f is None else f"{text}\nF={f:.3f}"
    y = 20
    for line in label.split("\n"):
        cv2.putText(out, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 200), 1, cv2.LINE_AA)
        y += 18
    return out


def _thumb(img: np.ndarray, max_w: int = 400) -> np.ndarray:
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Comparison figure (OpenCV-based, no matplotlib needed)
# ---------------------------------------------------------------------------

def make_comparison_figure(
    sample_id: str,
    bgr: np.ndarray,
    gts: list[np.ndarray],
    base_mask: np.ndarray,
    full_mask: np.ndarray,
    base_f: float,
    full_f: float,
    out_path: Path,
    max_w: int = 350,
) -> None:
    """Save a 1×4 comparison strip: original | GT | base | full."""
    panels = [
        _add_label(_thumb(bgr, max_w), "original"),
        _add_label(_thumb(_gt_mean(gts), max_w), "ground truth"),
        _add_label(_thumb(_mask_to_bgr(base_mask), max_w), "sauvola_base", base_f),
        _add_label(_thumb(_mask_to_bgr(full_mask), max_w), "+cleanup+norm", full_f),
    ]

    # Pad all panels to the same height
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        h = p.shape[0]
        if h < max_h:
            pad = np.full((max_h - h, p.shape[1], 3), 240, dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    # Add a 2-px white separator between panels
    sep = np.full((max_h, 2, 3), 200, dtype=np.uint8)
    strip = padded[0]
    for p in padded[1:]:
        strip = np.hstack([strip, sep, p])

    # Title bar
    title_h = 24
    title = np.full((title_h, strip.shape[1], 3), 50, dtype=np.uint8)
    cv2.putText(title, sample_id, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    final = np.vstack([title, strip])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Parameter sensitivity heatmap (requires matplotlib)
# ---------------------------------------------------------------------------

def make_param_sweep_figure(sweep_csv: str, out_path: Path) -> None:
    """Generate a parameter-sensitivity heatmap from param_sweep_refined.csv.
    Uses matplotlib if available, otherwise renders directly via OpenCV.
    """
    if not Path(sweep_csv).exists():
        print(f"[warn] {sweep_csv} not found — skipping param_sensitivity.png")
        return

    rows: list[dict] = []
    with open(sweep_csv) as fh:
        rows = list(csv.DictReader(fh))

    # The refined sweep has columns: window_rel, k, f_2018, f_2019, mean, min, selected
    metrics = [("f_2018", "DIBCO 2018"), ("f_2019", "DIBCO 2019"), ("min", "Worst-case (min)")]
    window_rels = sorted({float(r["window_rel"]) for r in rows})
    ks = sorted({float(r["k"]) for r in rows})

    if HAS_MPL:
        _render_heatmap_mpl(metrics, rows, window_rels, ks, out_path)
    else:
        _render_heatmap_cv2(metrics, rows, window_rels, ks, out_path)


def _render_heatmap_mpl(metrics, rows, window_rels, ks, out_path: Path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    fig.suptitle("Sauvola parameter sensitivity (F-measure refinement grid)", fontsize=11)

    for col, (field, title) in enumerate(metrics):
        ax = axes[0, col]
        Z = np.full((len(window_rels), len(ks)), np.nan)
        for r in rows:
            wi = window_rels.index(float(r["window_rel"]))
            ki = ks.index(float(r["k"]))
            Z[wi, ki] = float(r[field])

        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[min(ks) - 0.01, max(ks) + 0.01,
                               min(window_rels) - 0.005, max(window_rels) + 0.005],
                       vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)),
                       cmap="viridis")
        plt.colorbar(im, ax=ax, label="F-measure")

        # Mark the frozen parameter
        FROZEN_WR, FROZEN_K = 0.05, 0.15
        ax.scatter([FROZEN_K], [FROZEN_WR], color="red", marker="*",
                   s=200, zorder=5, label="selected (0.05, 0.15)")
        ax.set_xlabel("k")
        ax.set_ylabel("window_rel")
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name} (via matplotlib)")


def _render_heatmap_cv2(metrics, rows, window_rels, ks, out_path: Path):
    """Fallback: render the heatmap grid cleanly using OpenCV only."""
    cell_w, cell_h = 45, 45
    margin_l, margin_r, margin_t, margin_b = 60, 30, 60, 50
    plot_w = len(ks) * cell_w
    plot_h = len(window_rels) * cell_h
    panel_w = margin_l + plot_w + margin_r
    panel_h = margin_t + plot_h + margin_b

    panels = []
    for field, title in metrics:
        Z = np.full((len(window_rels), len(ks)), np.nan)
        for r in rows:
            wi = window_rels.index(float(r["window_rel"]))
            ki = ks.index(float(r["k"]))
            Z[wi, ki] = float(r[field])

        z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
        span = max(z_max - z_min, 1e-6)

        canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

        # Title
        cv2.putText(canvas, title, (margin_l, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(canvas, f"F: [{z_min:.3f}, {z_max:.3f}]", (margin_l, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Draw cells
        for r_idx, wr in enumerate(window_rels):
            y_cell = margin_t + (len(window_rels) - 1 - r_idx) * cell_h
            for c_idx, k in enumerate(ks):
                x_cell = margin_l + c_idx * cell_w
                val = Z[r_idx, c_idx]
                if np.isfinite(val):
                    norm = (val - z_min) / span
                    color_val = int(norm * 255)
                    one_px = np.array([[[color_val]]], dtype=np.uint8)
                    mapped = cv2.applyColorMap(one_px, cv2.COLORMAP_VIRIDIS)[0, 0]
                    cv2.rectangle(canvas, (x_cell, y_cell),
                                  (x_cell + cell_w - 1, y_cell + cell_h - 1),
                                  [int(c) for c in mapped], -1)

                    # Highlight the selected point (0.05, 0.15)
                    if abs(wr - 0.05) < 1e-4 and abs(k - 0.15) < 1e-4:
                        cv2.rectangle(canvas, (x_cell + 1, y_cell + 1),
                                      (x_cell + cell_w - 2, y_cell + cell_h - 2),
                                      (0, 0, 255), 2)
                        cv2.circle(canvas, (x_cell + cell_w // 2, y_cell + cell_h // 2),
                                   4, (0, 0, 255), -1)

        # Axes labels
        for c_idx, k in enumerate(ks):
            x = margin_l + c_idx * cell_w + 5
            cv2.putText(canvas, f"{k:.2f}", (x, margin_t + plot_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
        cv2.putText(canvas, "k", (margin_l + plot_w // 2, margin_t + plot_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for r_idx, wr in enumerate(window_rels):
            y = margin_t + (len(window_rels) - 1 - r_idx) * cell_h + cell_h // 2 + 4
            cv2.putText(canvas, f"{wr:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
        cv2.putText(canvas, "win_rel", (5, margin_t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        panels.append(canvas)

    combined = np.hstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"  saved {out_path.name} (via OpenCV fallback)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate failure-analysis figures for the report."
    )
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--results-csv", default="eval/results/final_eval.csv",
                    help="CSV with per-image F-measure results (ablation or final_eval).")
    ap.add_argument("--sauvola-config", default="configs/sauvola_config.json")
    ap.add_argument("--n-worst", type=int, default=3,
                    help="Number of worst images to show per dataset.")
    ap.add_argument("--channel", default="lab_l",
                    help="Channel for the 'best pipeline' column.")
    ap.add_argument("--param-sweep-csv", default="eval/results/param_sweep_refined.csv")
    ap.add_argument("--out-dir", default="eval/results/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_base_cfg(args.sauvola_config)
    cfg_base = _make_base_pipeline(base_cfg)
    cfg_full = _make_full_pipeline(base_cfg, args.channel)

    # Load palm_dev exclusion list
    splits_p = Path(args.splits)
    palm_dev: set[str] = set()
    if splits_p.exists():
        palm_dev = set(json.loads(splits_p.read_text()).get("palm_dev", []))

    # Build a lookup from image id → image_path across all datasets
    sample_map: dict[str, object] = {}
    for spec in args.dataset:
        ds_name, _, ds_path = spec.partition("=")
        for s in discover(ds_path, ds_name):
            sample_map[s.id] = s

    # Find worst images
    results_csv = args.results_csv
    if not Path(results_csv).exists():
        # Fallback to ablation.csv
        results_csv = "eval/results/ablation.csv"
        print(f"[info] {args.results_csv} not found, using {results_csv}")
    if not Path(results_csv).exists():
        print("[warn] No results CSV found. Skipping comparison figures.")
    else:
        worst = _load_worst(results_csv, args.n_worst)
        print(f"\nGenerating comparison figures ({args.n_worst} worst per dataset) …")
        for ds, ids in sorted(worst.items()):
            for rank, img_id in enumerate(ids, 1):
                if img_id in palm_dev:
                    continue
                s = sample_map.get(img_id)
                if s is None:
                    print(f"  [warn] {img_id} not found in provided datasets, skipped")
                    continue
                bgr, gts = load_sample(s)
                # Sauvola base
                base_stages = pipeline_run(bgr, cfg_base)
                base_mask = base_stages["cleaned"]
                base_f = float(np.mean([
                    _f_measure(base_mask, g) for g in gts
                ]))
                # Best pipeline
                full_stages = pipeline_run(bgr, cfg_full)
                full_mask = full_stages["cleaned"]
                full_f = float(np.mean([
                    _f_measure(full_mask, g) for g in gts
                ]))

                out_name = f"{ds}_worst{rank:02d}_{img_id.split('/')[-1]}.png"
                make_comparison_figure(
                    img_id, bgr, gts, base_mask, full_mask, base_f, full_f,
                    out_dir / out_name,
                )

    # Parameter sensitivity heatmap
    print("\nGenerating parameter sensitivity heatmap …")
    make_param_sweep_figure(args.param_sweep_csv, out_dir / "param_sensitivity.png")

    print(f"\nAll figures → {out_dir}/")


def _f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.ravel().astype(bool)
    gt = gt.ravel().astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


if __name__ == "__main__":
    main()

