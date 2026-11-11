"""Acquisition: image / ground-truth loading and dataset discovery.

Rules fixed for this project:
  * Ink is True. Ground truth is black-on-white, so ink = (gt < 128).
  * Every sample gets a globally unique ID: "<dataset>/<stem>"
    (e.g. "dibco2018/1", "dibco2019/1", "palm/Bangli-P9"), because DIBCO
    years reuse the same numeric file names.
  * Image and GT must have identical dimensions (checked on load).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# GT naming per dataset kind. Matching is by stem, extension-agnostic.
#   dibco: <stem>_gt.<ext>          palm: <stem>_GT1.<ext>, <stem>_GT2.<ext>
_GT_SUFFIX = re.compile(r"_(gt\d*)$", re.IGNORECASE)


@dataclass
class Sample:
    id: str
    dataset: str
    image_path: Path
    gt_paths: list[Path] = field(default_factory=list)


def load_color(path: Path) -> np.ndarray:
    """Load an image as BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def load_gt(path: Path) -> np.ndarray:
    """Load a ground-truth mask as a bool array, ink = True.

    Thresholded at 128 even for lossless files, so a stray non-binary GT
    cannot leak grey values into the metrics.
    """
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(f"could not read ground truth: {path}")
    return g < 128


def load_sample(sample: Sample) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (BGR image, [ink masks]) and assert matching dimensions."""
    img = load_color(sample.image_path)
    gts = [load_gt(p) for p in sample.gt_paths]
    for p, g in zip(sample.gt_paths, gts):
        if g.shape != img.shape[:2]:
            raise ValueError(
                f"{sample.id}: image {img.shape[:2]} and GT {g.shape} differ ({p.name})"
            )
    return img, gts


def discover(root: str | Path, dataset: str) -> list[Sample]:
    """Find samples under `root`, auto-detecting the folder layout.

    * Separated layout: <root>/images/ plus <root>/gt/ (DIBCO) or
      <root>/gt1/ and <root>/gt2/ (palm leaf).
    * Flat layout: everything mixed in <root> itself.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "images").is_dir():
        return _discover_separated(root, dataset)
    return _discover_flat(root, dataset)


_GT_DIR = re.compile(r"^gt\d*$", re.IGNORECASE)


def _discover_separated(root: Path, dataset: str) -> list[Sample]:
    """<root>/images/N.ext with GTs in <root>/gt*/ named N.ext, N_gt.ext, N_GT1.ext ..."""
    gt_dirs = sorted(d for d in root.iterdir() if d.is_dir() and _GT_DIR.match(d.name))
    if not gt_dirs:
        raise FileNotFoundError(f"{root}: has images/ but no gt/ (or gt1/, gt2/) folder")

    # index every GT folder: base-stem -> path
    index: list[dict[str, Path]] = []
    for d in gt_dirs:
        m: dict[str, Path] = {}
        for p in d.iterdir():
            if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                m[_GT_SUFFIX.sub("", p.stem)] = p
        index.append(m)

    samples: list[Sample] = []
    for p in sorted((root / "images").iterdir(), key=lambda q: q.stem):
        if p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("."):
            continue
        gts = [m[p.stem] for m in index if p.stem in m]
        if len(gts) < len(gt_dirs):
            missing = [d.name for d, m in zip(gt_dirs, index) if p.stem not in m]
            print(f"[warn] {dataset}/{p.stem}: missing GT in {', '.join(missing)}")
        if not gts:
            print(f"[warn] no ground truth for {p.name}, skipped")
            continue
        samples.append(Sample(id=f"{dataset}/{p.stem}", dataset=dataset,
                              image_path=p, gt_paths=gts))
    return samples


def _discover_flat(root: Path, dataset: str) -> list[Sample]:
    """Find (image, GT...) samples in a flat folder.

    An image is any image file whose stem does NOT end in _gt / _GT1 / _GT2.
    Its GTs are files named <stem>_gt.* (DIBCO) or <stem>_GT1.*, <stem>_GT2.*
    (palm leaf). Images without any GT are skipped with a warning.
    """
    files = [p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p)

    samples: list[Sample] = []
    for p in sorted(files, key=lambda q: q.stem):
        if _GT_SUFFIX.search(p.stem):
            continue                                  # this is a GT file
        gts: list[Path] = []
        for stem, paths in by_stem.items():
            m = _GT_SUFFIX.search(stem)
            if m and stem[: m.start()] == p.stem:
                gts.extend(paths)
        gts = sorted(gts, key=lambda q: q.stem.lower())
        if not gts:
            print(f"[warn] no ground truth for {p.name}, skipped")
            continue
        samples.append(Sample(id=f"{dataset}/{p.stem}", dataset=dataset,
                              image_path=p, gt_paths=gts))
    return samples
