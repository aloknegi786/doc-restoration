"""QC script: visualise ROI masks and Otsu seed overlays for a sample of images.

Purpose: inspect the ROI stage *before* any test numbers are produced.
Only palm dev images and a few tuning images are shown here; test sets
are never touched.

Output layout (all written to --out-dir):
    {stem}_roi.jpg    original + green ROI fill + green contour
    {stem}_seed.jpg   original + red Otsu seed fill

Usage (from project root, in .venv):
    python -m eval.qc_roi \\
        --dataset dibco2018=data/dibco2018 \\
        --dataset dibco2019=data/dibco2019 \\
        --dataset palm=data/palm_split \\
        --splits data/splits.json \\
        --n-tuning 2 \\
        --out-dir eval/results/qc_roi

The script shows:
  * All 5 palm dev images.
  * Up to --n-tuning images from each DIBCO tuning dataset (default 2).
  * Never any test-set images.

For each image it prints one line:
    dataset/stem  ROI coverage=XX.X%  Otsu seed coverage=XX.X%
so you can quickly spot pathological cases without opening every file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover, load_sample, to_gray
from docbin.roi import detect_roi, draw_mask_overlay, draw_roi_overlay
from docbin.segmentation import otsu_binarize


def coverage_pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.sum()) / mask.size


def process_image(
    sample_id: str,
    bgr: np.ndarray,
    out_dir: Path,
    roi_kwargs: dict,
) -> None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    roi_mask = detect_roi(bgr, **roi_kwargs)
    seed_mask = otsu_binarize(gray)

    roi_cov = coverage_pct(roi_mask)
    seed_cov = coverage_pct(seed_mask)

    # Flag pathological cases
    roi_flag = "  *** FULL FRAME FALLBACK ***" if roi_cov > 94.9 else ""
    seed_flag = "  *** OVERMARKS INK ***" if seed_cov > 50.0 else ""
    print(
        f"  {sample_id:<40}  ROI={roi_cov:5.1f}%{roi_flag}"
        f"   seed={seed_cov:5.1f}%{seed_flag}"
    )

    stem = sample_id.replace("/", "_")

    # ROI overlay: green fill + contour
    roi_vis = draw_roi_overlay(bgr, roi_mask,
                               fill_color=(0, 200, 0),
                               contour_color=(0, 255, 0),
                               alpha=0.25, contour_thickness=2)
    cv2.imwrite(str(out_dir / f"{stem}_roi.jpg"), roi_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Otsu seed overlay: red fill
    seed_vis = draw_mask_overlay(bgr, seed_mask,
                                  fill_color=(0, 0, 200),
                                  alpha=0.35)
    cv2.imwrite(str(out_dir / f"{stem}_seed.jpg"), seed_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


def select_samples(args, splits: dict) -> list[tuple[str, Path, str]]:
    """Return a list of (sample_id, image_path, dataset_name) to process.

    Rule:
      - For palm: only the palm_dev ids in splits.json.
      - For every other dataset: up to --n-tuning images from the tuning split.
      - Test-set ids (dibco2017, palm test) are never included.
    """
    palm_dev_ids: set[str] = set(splits.get("palm_dev", []))
    tuning_ids: set[str] = set()
    for fold in splits.get("tuning", {}).values():
        tuning_ids.update(fold)

    selected: list[tuple[str, Path, str]] = []

    for spec in args.dataset:
        name, _, path = spec.partition("=")
        all_samples = discover(path, name)

        if name == "palm":
            # Only the 5 dev images
            for s in all_samples:
                if s.id in palm_dev_ids:
                    selected.append((s.id, s.image_path, name))
        else:
            # Tuning images only, up to n_tuning per dataset
            count = 0
            for s in sorted(all_samples, key=lambda x: x.id):
                if s.id in tuning_ids and count < args.n_tuning:
                    selected.append((s.id, s.image_path, name))
                    count += 1

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Save ROI and Otsu seed overlays for a visual QC check."
    )
    ap.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=DIR",
        help="Dataset name and root directory, e.g. dibco2018=data/dibco2018"
    )
    ap.add_argument(
        "--splits", default="data/splits.json",
        help="Path to splits.json (selects palm_dev and tuning ids)."
    )
    ap.add_argument(
        "--n-tuning", type=int, default=2, metavar="N",
        help="How many tuning images to include per DIBCO dataset (default 2)."
    )
    ap.add_argument(
        "--out-dir", default="eval/results/qc_roi",
        help="Directory for overlay images."
    )
    # ROI tuning knobs (for experimenting with the visual output)
    ap.add_argument("--margin-frac", type=float, default=0.015)
    ap.add_argument("--close-frac", type=float, default=0.04)
    ap.add_argument("--downsample", type=int, default=4)
    args = ap.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    splits = json.loads(splits_path.read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_kwargs = {
        "margin_frac": args.margin_frac,
        "close_frac": args.close_frac,
        "downsample": args.downsample,
    }

    samples = select_samples(args, splits)
    if not samples:
        print("No samples found. Check --dataset paths and --splits.")
        sys.exit(1)

    print(f"Processing {len(samples)} images → {out_dir}/\n")
    print(f"{'sample id':<42}  {'ROI':>9}   {'seed':>9}")
    print("-" * 70)

    for sample_id, img_path, ds_name in samples:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [warn] could not read {img_path}, skipped")
            continue
        process_image(sample_id, bgr, out_dir, roi_kwargs)

    print(f"\nDone. {len(samples) * 2} images written to {out_dir}/")
    print(
        "\nWhat to look for:\n"
        "  _roi.jpg  — green fill should cover the page/leaf, NOT the grey surround.\n"
        "              If it shows 'FULL FRAME FALLBACK', the detection found nothing.\n"
        "  _seed.jpg — red fill should cover ink strokes, NOT most of the image.\n"
        "              Seed coverage > 50% on palm = broken stroke-width estimate."
    )


if __name__ == "__main__":
    main()

