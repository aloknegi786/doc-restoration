"""Split a flat palm-leaf folder into three folders: images/, gt1/, gt2/.

Input folder contains, per manuscript:
    NAME.jpg   NAME_GT1.bmp   NAME_GT2.bmp

Usage (run from anywhere; paths may be relative or absolute):
    python tools/segregate_palm.py --src "path/to/train-50" --dst data/palm_split
    python tools/segregate_palm.py --src ... --dst ... --dry-run     # preview only
    python tools/segregate_palm.py --src ... --dst ... --move        # move, don't copy

Result:
    <dst>/images/NAME.jpg
    <dst>/gt1/NAME_GT1.bmp
    <dst>/gt2/NAME_GT2.bmp

File names are kept unchanged. Existing files in <dst> are never overwritten.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
GT_RE = re.compile(r"^(?P<name>.+)_GT(?P<n>[12])$", re.IGNORECASE)


def classify(src: Path):
    images, gt1, gt2, other = {}, {}, {}, []
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.name.startswith("."):      # skips .DS_Store etc.
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            other.append(p)
            continue
        m = GT_RE.match(p.stem)
        if m:
            (gt1 if m.group("n") == "1" else gt2)[m.group("name")] = p
        else:
            images[p.stem] = p
    return images, gt1, gt2, other


def transfer(files, dest: Path, move: bool, dry: bool) -> int:
    done = 0
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
    for p in files:
        target = dest / p.name
        if target.exists():
            print(f"  [skip] exists: {target}")
            continue
        if not dry:
            (shutil.move if move else shutil.copy2)(str(p), str(target))
        done += 1
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder with all palm files mixed together")
    ap.add_argument("--dst", required=True, type=Path, help="output folder (images/, gt1/, gt2/ created inside)")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"error: source folder not found: {args.src}")
        return 1
    if args.src.resolve() == args.dst.resolve():
        print("error: --dst must be different from --src")
        return 1

    images, gt1, gt2, other = classify(args.src)
    print(f"found: {len(images)} images, {len(gt1)} GT1, {len(gt2)} GT2"
          + (f", {len(other)} unrelated files ignored" if other else ""))

    # consistency report: every image should have both GTs, and no GT should be orphaned
    problems = 0
    for name in sorted(images):
        missing = [t for t, d in (("GT1", gt1), ("GT2", gt2)) if name not in d]
        if missing:
            problems += 1
            print(f"  [warn] {name}: missing {', '.join(missing)}")
    for label, d in (("GT1", gt1), ("GT2", gt2)):
        for name in sorted(set(d) - set(images)):
            problems += 1
            print(f"  [warn] orphan {label} without base image: {d[name].name}")

    verb = "would " + ("move" if args.move else "copy") if args.dry_run else ("moved" if args.move else "copied")
    n_i = transfer(images.values(), args.dst / "images", args.move, args.dry_run)
    n_1 = transfer(gt1.values(), args.dst / "gt1", args.move, args.dry_run)
    n_2 = transfer(gt2.values(), args.dst / "gt2", args.move, args.dry_run)
    print(f"{verb}: {n_i} images -> {args.dst/'images'}, {n_1} -> {args.dst/'gt1'}, {n_2} -> {args.dst/'gt2'}")
    if problems:
        print(f"{problems} problem(s) reported above; fix them before running evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
