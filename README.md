# Document Restoration & Binarization

Classical image-processing pipeline that converts degraded historical document
images (uneven illumination, bleed-through, stains, faded ink) into clean
binary images (ink vs. paper), ending immediately before OCR / recognition.

---

## Pipeline stages

```
BGR image
  ↓ channel selection     (gray | red | LAB-L | PCA)
  ↓ ROI detection         (Otsu on LAB-L, largest CC, fill holes, erode border)
  ↓ background normalise  (morphological closing → division, ROI-masked)
  ↓ CLAHE                 (from scratch, 8×8 tile grid, clip=2.0)
  ↓ Sauvola binarize      (integral-image implementation, window_rel=0.05, k=0.15)
  ↓ small-component cleanup
  ↓ binary mask (ink = True)
```

All stages are pure functions (`ndarray + params → ndarray`).  Every
parameter lives in one `PipelineConfig` dataclass.  The pipeline returns
all intermediate stages, which powers both the ablation study and the API's
`/binarize/stages` route.

---

## Repository layout

```
docbin/           Core algorithm library
  io.py             Image loading, dataset discovery
  segmentation.py   Otsu, Niblack, Sauvola (integral images)
  scale.py          Stroke-width estimator (skeleton + distance transform)
  roi.py            ROI detection (LAB-L Otsu, morphological post-processing)
  colour.py         Channel selection (gray, red, LAB-L, PCA)
  normalise.py      Background normalisation (closing-based estimation + division)
  clahe.py          CLAHE from scratch (tile LUTs + bilinear interpolation)
  cleanup.py        Small-component removal
  config.py         PipelineConfig dataclass
  pipeline.py       End-to-end chain returning all intermediates

eval/             Evaluation scripts
  metrics.py        Precision, Recall, F-Measure, Dice, PSNR
  make_splits.py    Generates data/splits.json (frozen, seed=0)
  run_baseline.py   Otsu + Sauvola baselines on all datasets
  param_sweep.py    Coarse + refined cross-year parameter tuning
  calibrate_scale.py  Stroke-width factor calibration
  diagnostics.py    Seed-mask ink fraction + Wilcoxon tests
  ablation.py       Stage + channel ablation study (tuning sets only)
  final_eval.py     One-shot test-set evaluation (run once, frozen pipeline)
  qc_roi.py         QC overlays for ROI stage inspection

backend/          FastAPI service
  main.py           POST /binarize, POST /binarize/stages, GET /health

data/             Datasets (not committed — see Data section)
configs/          Frozen parameter configs (sauvola_config.json, scale_config.json)
tests/            pytest test suite (57 tests)
report/           Checkpoint report and final report
Dockerfile        Single-container backend image
```

---

## Quickstart with Docker (Self-Contained)

The easiest way to run the project. No Python or environment setup required:

```bash
# 1. Build the Docker container
docker build -t doc-restoration .

# 2. Run the application
docker run -p 8000:8000 doc-restoration
```

Open **[http://localhost:8000/](http://localhost:8000/)** in your browser. 
Sample degraded documents (DIBCO bleed-through, water stains, and palm leaf manuscripts) are embedded in the container and can be tested with a single click.

---

## Local Setup (without Docker)

```bash
# 1. Clone and create venv
git clone <repo> && cd doc-restoration
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -q

# 4. Prepare data  (see Data section below)

# 5. Run the baseline (reproduces Table 6 of the report)
python -m eval.run_baseline \
    --dataset dibco2017=data/dibco2017 \
    --dataset dibco2018=data/dibco2018 \
    --dataset dibco2019=data/dibco2019 \
    --dataset palm=data/palm_split \
    --splits data/splits.json

# 6. Run the full ablation (tuning sets only)
python -m eval.ablation \
    --dataset dibco2018=data/dibco2018 \
    --dataset dibco2019=data/dibco2019 \
    --splits data/splits.json \
    --out eval/results/ablation.csv

# 7. Final test-set evaluation (run ONCE after pipeline is frozen)
python -m eval.final_eval \
    --dataset dibco2017=data/dibco2017 \
    --dataset palm=data/palm_split \
    --splits data/splits.json \
    --channel lab_l
```

---

## Web UI & API Server

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn backend.main:app --reload --port 8000
```

- **Interactive Web Demo:** Visit [http://localhost:8000/](http://localhost:8000/) for the drag-and-drop comparison UI (live side-by-side preview, pipeline parameter controls, and direct downloads).
- **Swagger Documentation:** Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive OpenAPI documentation.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status + active config |
| `POST` | `/binarize` | Upload image → PNG binary mask |
| `POST` | `/binarize/stages` | Upload image → ZIP with all stage PNGs |

Query parameters on `/binarize` and `/binarize/stages`:
`channel` (gray/red/lab_l/pca), `use_roi`, `use_normalise`, `use_clahe`, `use_cleanup`.

---

## Docker

```bash
docker build -t doc-restoration .
docker run -p 8000:8000 doc-restoration
```

---

## Data

Expected folder layout (not committed — obtain datasets separately):

```
data/
  dibco2017/   images/1.bmp … 20.bmp    gt/1_gt.bmp … 20_gt.bmp
  dibco2018/   images/1.bmp … 10.bmp    gt/1_gt.bmp … 10_gt.bmp
  dibco2019/   images/1.bmp … 10.bmp    gt/1_gt.bmp … 10_gt.bmp
  palm_split/  images/*.jpg             gt1/*_GT1.bmp   gt2/*_GT2.bmp
  splits.json  (frozen, seed=0 — do not regenerate)
```

To prepare palm leaf data from a mixed folder:
```bash
python -m tools.segregate_palm <mixed_dir> data/palm_split
```

---

## Frozen parameters (do not retune)

| Parameter | Value | Source |
|---|---|---|
| `window_rel` | 0.05 | `configs/sauvola_config.json` |
| `k` | 0.15 | `configs/sauvola_config.json` |
| `factor` | 7.5 | `configs/scale_config.json` |
| `floor` | 15 | `configs/scale_config.json` |
| Splits seed | 0 | `data/splits.json` |

---

## Implementation notes

- **Implemented from scratch and validated against scikit-image / OpenCV:**
  Otsu, Niblack, Sauvola (integral images), CLAHE, all metrics.
- **Library calls used for:** image I/O, colour conversion, morphology,
  skeletonisation.
- **Niblack sign convention:** this project uses the original `T = m + k·s`
  (k negative); scikit-image uses `T = m − k·s`, so skimage `k=+0.2`
  equals `k=−0.2` here.
- F-Measure and Dice are mathematically identical for binary masks
  (`2·TP / (2·TP + FP + FN)`); both are reported per the brief, with this
  equivalence stated explicitly.

