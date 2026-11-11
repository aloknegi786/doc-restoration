"""Write data/splits.json: the fixed, reproducible experimental design.

Design (agreed):
  * TUNING     : DIBCO 2018 + 2019, cross-year folds
                   fold A: tune on 2018, validate on 2019
                   fold B: tune on 2019, validate on 2018
  * TEST       : DIBCO 2017 (all), palm leaf (all except dev)
  * PALM DEV   : N palm leaf images chosen at random (fixed seed), used only
                 to sanity-check the ROI stage and NEVER reported in results.

Usage:
    python -m eval.make_splits --dibco2018 data/dibco2018 --dibco2019 data/dibco2019 \
        --dibco2017 data/dibco2017 --palm data/palm --n-dev 5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.io import discover


def ids(path: str, name: str) -> list[str]:
    return sorted(s.id for s in discover(path, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dibco2018", required=True)
    ap.add_argument("--dibco2019", required=True)
    ap.add_argument("--dibco2017", required=True)
    ap.add_argument("--palm", required=True)
    ap.add_argument("--n-dev", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/splits.json")
    args = ap.parse_args()

    d18, d19 = ids(args.dibco2018, "dibco2018"), ids(args.dibco2019, "dibco2019")
    d17, palm = ids(args.dibco2017, "dibco2017"), ids(args.palm, "palm")

    if args.n_dev >= len(palm):
        raise SystemExit("n-dev must be smaller than the number of palm images")
    dev = sorted(random.Random(args.seed).sample(palm, args.n_dev))

    splits = {
        "seed": args.seed,
        "tuning": {"dibco2018": d18, "dibco2019": d19},
        "folds": [
            {"name": "A", "tune": d18, "validate": d19},
            {"name": "B", "tune": d19, "validate": d18},
        ],
        "test": {"dibco2017": d17, "palm": [p for p in palm if p not in dev]},
        "palm_dev": dev,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2))
    print(f"tuning: {len(d18)}+{len(d19)}  test: dibco2017={len(d17)} "
          f"palm={len(palm) - len(dev)}  palm_dev={len(dev)}  -> {out}")


if __name__ == "__main__":
    main()
