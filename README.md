# Document Restoration & Binarization

A high-performance classical image-processing pipeline and web service that restores degraded historical document images (uneven illumination, bleed-through, water stains, fading) and converts them into clean binary images (ink vs. paper).

Includes a modern interactive browser interface (live comparison slider, sample loader, parameter inspection) and a FastAPI backend with Swagger docs.

---

## Pipeline Stages

```
Input Image (BGR)
  │
  ├── 1. Channel Selection      [gray | red | lab_l | pca]
  ├── 2. ROI Detection          [LAB-L Otsu, largest connected component, hole fill, margin trim]
  ├── 3. Background Normalise   [Morphological closing background estimation + division]
  ├── 4. CLAHE                  [Contrast-limited adaptive histogram equalization]
  ├── 5. Sauvola Thresholding   [Fast integral-image adaptive local binarization]
  └── 6. Morphological Cleanup  [Small-component noise removal]
  │
Output Binary Mask (Ink = True / Black, Paper = False / White)
```

Every stage is a pure function (`ndarray + config → ndarray`). All parameters are centrally managed in `PipelineConfig`, and intermediate representations can be extracted for inspection via the API.

---

## Project Structure

Every file in the codebase serves a clear and distinct purpose:

```
├── backend/
│   ├── main.py             # FastAPI service (API routes, sample serving, stage packaging)
│   └── static/
│       └── index.html      # Responsive web UI with instant sample loading & comparison
├── docbin/                 # Core image processing library
│   ├── __init__.py         # Package entry point (exports PipelineConfig, run, binarize)
│   ├── clahe.py            # Custom CLAHE algorithm with bilinear interpolation
│   ├── cleanup.py          # Small-component morphological noise removal
│   ├── colour.py           # Multi-space channel selector (grayscale, red, LAB-L, PCA)
│   ├── config.py           # PipelineConfig dataclass and serialization helpers
│   ├── normalise.py        # Background normalisation via morphological closing & division
│   ├── pipeline.py         # End-to-end pipeline orchestrator returning all intermediates
│   ├── roi.py              # Document ROI boundary detection & visual overlay generator
│   └── segmentation.py     # Integral-image Otsu and Sauvola local adaptive thresholding
├── configs/
│   └── pipeline_config.json# Default active pipeline parameters loaded on startup
├── sample_images/          # Benchmark historical samples for one-click testing in UI
├── tests/                  # Automated pytest suite
│   ├── test_backend.py     # API endpoint integration and error handling tests
│   ├── test_pipeline.py    # Unit tests for individual stages and pipeline execution
│   └── test_segmentation.py# Tests for Otsu, Sauvola, and integral image math
├── Dockerfile              # Production container build
├── pyproject.toml          # Package configuration and test settings
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Testing & development dependencies
└── .github/workflows/ci.yml# Continuous integration workflow
```

---

## Quickstart with Docker

Run the entire application in a container with zero local environment setup:

```bash
# 1. Build the Docker image
docker build -t doc-restoration .

# 2. Run container
docker run -p 8000:8000 doc-restoration
```

Open **[http://localhost:8000/](http://localhost:8000/)** in your browser. Sample documents (stains, bleed-through, palm-leaf manuscripts) can be tested instantly using the sample chips in the UI.

---

## Local Setup

### 1. Environment Setup

```bash
# Clone and enter the repository
git clone <repo> && cd doc-restoration

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Application

```bash
uvicorn backend.main:app --reload --port 8000
```

- **Interactive UI:** Open [http://localhost:8000/](http://localhost:8000/)
- **Swagger Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run pytest suite
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Web UI |
| `GET` | `/health` | Service health status and loaded `PipelineConfig` |
| `GET` | `/samples/{filename}` | Retrieve sample historical document images |
| `POST` | `/binarize` | Upload image $\rightarrow$ returns binary mask PNG (Ink = Black) |
| `POST` | `/binarize/stages` | Upload image $\rightarrow$ returns ZIP archive with PNGs of all intermediate stages |

### Query Parameters

On both `/binarize` and `/binarize/stages`, stages can be customized dynamically:
- `channel`: `gray` (default), `red`, `lab_l`, `pca`
- `use_roi`: `true` / `false`
- `use_normalise`: `true` / `false`
- `use_clahe`: `true` / `false`
- `use_cleanup`: `true` / `false`

### Stages Archive Output

When requesting `/binarize/stages`, the downloaded `.zip` contains:
1. `00_original.png` — Original uploaded colour image
2. `01_channel.png` — Converted single-channel image
3. `02_roi.png` — Detected document ROI overlay
4. `03_normalised.png` — Background-corrected image
5. `04_clahe.png` — CLAHE contrast-enhanced image
6. `05_binarized.png` — Raw Sauvola binary output
7. `06_cleaned.png` — Final cleaned binary mask

---

## Configuration

The backend loads default settings from `configs/pipeline_config.json`:

```json
{
  "channel": "gray",
  "use_roi": false,
  "roi_margin_frac": 0.015,
  "roi_close_frac": 0.04,
  "roi_downsample": 4,
  "use_normalise": true,
  "bg_sigma_frac": 0.1,
  "bg_downsample": 4,
  "use_clahe": false,
  "clahe_grid": [8, 8],
  "clahe_clip": 2.0,
  "window_rel": 0.05,
  "k": 0.15,
  "R": 128.0,
  "use_cleanup": true,
  "min_component_size": 15
}
```
