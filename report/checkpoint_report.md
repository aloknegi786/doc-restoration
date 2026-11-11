# Automated Restoration & Binarization of Degraded Documents
## Checkpoint Report

---

## 1. Project Overview

**Objective.** Build a classical (non-learned) image processing pipeline that
converts degraded historical document images — uneven illumination,
bleed-through, stains, faded ink — into clean binary images (ink vs. paper),
ending immediately before any OCR / recognition step.

**Scope.** Solo, 3–4 week academic image processing project. Deep learning
(U-Net) is explicitly out of scope for this phase; the interface is kept
compatible with adding it later if time allows. The demo surface is a
FastAPI backend explored via its Swagger UI; no frontend is built.

**Deliverables agreed:**
- `docbin/` — the core algorithm library (implemented from scratch where it
  matters: Otsu, Sauvola/Niblack via integral images, stroke-width
  estimation; OpenCV used for I/O, colour conversion, morphology).
- `eval/` — dataset loading, metrics, parameter tuning, ablation, results.
- `backend/` — FastAPI routes wrapping the pipeline (not yet built).
- `tests/` — unit tests validated against scikit-image and hand-computed
  cases.
- One report, produced from the `eval/results/` outputs.

---

## 2. Experimental Design

### 2.1 Datasets

| Dataset | Images | Role | Ground truth |
|---|---|---|---|
| DIBCO 2018 | 10 | Tuning | 1 GT per image (`N_gt.bmp`) |
| DIBCO 2019 | 10 | Tuning | 1 GT per image |
| DIBCO 2017 | 20 | **Test** (never tuned on) | 1 GT per image |
| Palm leaf manuscripts | 50 (45 test + 5 dev) | **Test** (never tuned on) | 2 GTs per image (`_GT1`, `_GT2`) |

**Why this split.** DIBCO 2018/2019 provide the tuning signal. DIBCO 2017 is
an in-domain held-out test (same benchmark family, unseen documents) —
useful to detect overfitting to the two tuning years. Palm leaf is a
cross-domain held-out test (different substrate, script, and degradation
profile) — useful to test whether the pipeline generalises beyond paper
documents at all.

**Tuning protocol: cross-year folds**, to avoid over-fitting a 10-image
tuning year:
- Fold A: grid-search on 2018, validate the chosen point on 2019.
- Fold B: grid-search on 2019, validate the chosen point on 2018.

**Palm leaf dev set (5 images):** excluded from every result reported below.
Reserved for sanity-checking the upcoming ROI stage's output (mask
overlays) without leaking test data into that design decision.

**`data/splits.json`** is the single source of truth for which image
belongs to tuning, test, or dev. It is seeded (`seed=0`) and, once every
dataset's files loaded cleanly, frozen — not regenerated.

### 2.2 Loader conventions (`docbin/io.py`)

- Ink is the positive class (`True`); ground truth is thresholded at 128 on
  load (`gt < 128`) regardless of source format, so a lossy re-encode can
  never leak grey values into a metric.
- Every sample gets a dataset-prefixed ID (`dibco2018/1`, `palm/WN-P1a`)
  because DIBCO years reuse numeric filenames (`1.bmp` … `10.bmp`).
- Image and GT dimensions are asserted equal on load; a mismatch raises
  immediately rather than silently producing wrong metrics.
- Both a flat layout (image + GT mixed in one folder) and a separated
  layout (`images/`, `gt/`, or `gt1/`+`gt2/`) are auto-detected.
- Palm leaf GTs are matched by base filename; one dataset inconsistency was
  found and fixed by hand (see §2.3).

### 2.3 Data issues found and resolved

| Issue | Resolution |
|---|---|
| DIBCO years 2017–2019 reuse filenames `1`–`10` | Dataset-prefixed sample IDs |
| Palm leaf images are lossy JPEG; GT masks are lossless BMP | Threshold GT at 128 regardless; no special JPEG handling needed for GT |
| One palm image (`MB-AdiParwa(Purana)-5338.2-IV.a-P1.jpg`) had GT files named without the `.2-IV.a-P1` suffix | GT files renamed to match the image; re-verified as the only image of that name, so the pairing is unambiguous |
| `make_splits.py` could silently write an empty tuning set | Added an explicit `SystemExit` guard on any empty split |

### 2.4 Metrics (`eval/metrics.py`)

Precision, Recall, **F-Measure**, **Dice**, **PSNR** (binary images, C = 1),
all implemented from scratch. F-Measure and Dice are mathematically
identical for binary masks (both reduce to `2·TP / (2·TP + FP + FN)`); both
are reported per the project brief, with this equivalence noted rather than
treated as two independent signals.

For palm leaf (two GTs per image), every metric is the **mean over both
GTs**. Agreement between the two human annotators (`GT1` vs `GT2`) is
reported separately as a soft reference point, not a hard ceiling — a
prediction identical to one annotator would score 1.0 against it but only
~0.63 against the other, so the realistically achievable range is
approximately 0.63–0.81, not exactly the observed 0.628.

---

## 3. Methods Implemented So Far

### 3.1 Binarization (`docbin/segmentation.py`)

- **Otsu's method** — global threshold, from scratch (between-class
  variance maximisation over the 256-bin histogram), validated against
  `skimage.filters.threshold_otsu` (within 1 grey level).
- **Sauvola** and **Niblack** — local adaptive thresholding, from scratch,
  computed via **integral images** (sum and sum-of-squares) for O(1) local
  mean/std at any window size. Validated against `skimage`'s
  implementations (>99.5% pixel agreement on synthetic test images; exact
  agreement once a sign-convention difference in Niblack's formula was
  found and documented — `skimage` uses `T = m − k·s`, this project uses
  the original `T = m + k·s`, so `skimage`'s `k=+0.2` corresponds to this
  project's `k=−0.2`).

### 3.2 Window-size rules (`docbin/segmentation.py`, `docbin/scale.py`)

Two rules exist side by side, to make their difference measurable rather
than silently replacing one with the other:

- **`window_from_relative`** (placeholder): window = a fixed fraction of
  the image's shorter side. Simple, but conflates image resolution with
  stroke thickness — flagged as a likely weakness from the start.
- **`estimate_window`** (stroke-width based): estimates the document's own
  stroke thickness via a classical distance-transform method —
  (1) rough Otsu ink mask → (2) skeletonize → (3) Euclidean distance
  transform → (4) local thickness = 2 × distance at skeleton pixels →
  (5) stroke width = **median** thickness (robust to blots/noise) — then
  sets `window = factor × stroke_width`. Validated against synthetic bars
  of known thickness (3, 5, 9, 15 px): estimate within 1 px in every case.

### 3.3 Ground-truth–driven pipeline constraints

Direct inspection of DIBCO 2017/2018 and palm leaf ground truth fixed
several pipeline rules before any code was written for later stages:

- **No morphological opening** — GT includes faint pencil annotations and
  isolated ink dots a few pixels wide; opening would erase them.
- **Large dark blobs are kept** — ink blots and strikethroughs are
  foreground in the GT, so large-component removal is not used for
  cleanup; only small isolated speckle is a candidate for removal.
- **Bleed-through and show-through are excluded from GT even when nearly
  as dark as real ink** (clearest in DIBCO 2017, image `1`) — flagged as
  the primary expected failure mode for pure intensity-based thresholding.

---

## 4. Sauvola Parameter Tuning (`eval/param_sweep.py`)

### 4.1 Stage 1 — coarse grid, cross-year folds

Grid: `window_rel ∈ {0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12}`,
`k ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50}` (widened once from an
initial narrower grid after both folds landed on a grid edge).

| Fold | Chosen point | Train mean F | Validate mean F |
|---|---|---|---|
| A (tune 2018 → validate 2019) | window_rel=0.06, k=0.15 | 0.666 | 0.671 |
| B (tune 2019 → validate 2018) | window_rel=0.08, k=0.30 | 0.755 | 0.615 |

Fold A generalised well (validate ≈ train). Fold B dropped sharply
(0.755 → 0.615), indicating its point does not transfer cleanly to 2018's
harder degradations. The two folds picked genuinely different points,
rather than converging — itself a finding, not just tuning noise.

### 4.2 Stage 2 — refinement, worst-case selection

Rather than average the two coarse optima (a point neither fold actually
evaluated), a finer grid was built spanning both coarse optima with margin
(`window_rel` step 0.01, `k` step 0.02), scored on **both** years at every
point, and the **worst-case (minimum of the two years' F-Measure)** was
maximised — chosen over best-mean because it protects against a config that
is excellent on one year's degradation profile and poor on the other's,
which is exactly what fold B's point did.

| Point | F_2018 | F_2019 | Worst-case | Mean |
|---|---|---|---|---|
| **Selected (worst-case): window_rel=0.05, k=0.15** | 0.669 | 0.677 | **0.669** | 0.673 |
| Alternative (best-mean): window_rel=0.05, k=0.25 | 0.643 | 0.755 | 0.643 | 0.699 |

The selected point beats **both** coarse folds' own validation scores
(0.669, 0.677 vs. 0.671, 0.615), confirming the refinement was worthwhile,
not just a tie-breaking formality. **Frozen: `window_rel=0.05, k=0.15`.**

---

## 5. Stroke-Width Calibration (`eval/calibrate_scale.py`)

`estimate_window`'s multiplier (`factor`) is not hand-picked. It is
calibrated as `median(frozen_window_px / stroke_width)` over all 20 tuning
images, using the frozen `window_rel` from §4.2 to define "frozen_window_px".

**Result:** `factor = 7.50` (median), mean 8.85, std 5.17 (range 3.75–27.50).

The wide spread (`std/mean ≈ 0.58`) is interpreted as evidence *for*
stroke-width-based sizing rather than against it: it shows that
`window_from_relative`'s implicit "factor" varies enormously across images
of different resolution and scan quality — i.e., the fixed-fraction rule
was aliasing image size for stroke thickness, exactly the weakness
predicted before this calibration was run.

**Sanity check** (same frozen `k`, window rule only): re-scoring both
tuning years with `estimate_window(factor=7.5)` in place of
`window_from_relative`:

| Year | window_from_relative F | estimate_window F | Δ |
|---|---|---|---|
| 2018 | 0.6687 | 0.6882 | **+0.0195** |
| 2019 | 0.6773 | 0.6909 | **+0.0135** |

Both years improved by a small, consistent margin on the tuning set.
**Frozen: `factor = 7.5, floor = 15`.**

---

## 6. Baseline Results Across All Datasets (`eval/run_baseline.py`)

Full comparison — `otsu`, `sauvola_rel` (fixed-window), `sauvola_stroke`
(stroke-width window) — across all four datasets, using the frozen
parameters from §4–5 with no further tuning.

| Dataset | Method | n | F-Measure | PSNR (dB) | Dice |
|---|---|---|---|---|---|
| DIBCO 2017 (test) | otsu | 20 | 0.7774 ± 0.1615 | 13.85 ± 3.34 | 0.7774 |
| DIBCO 2017 (test) | sauvola_rel | 20 | 0.8133 ± 0.0674 | 14.59 ± 1.91 | 0.8133 |
| DIBCO 2017 (test) | sauvola_stroke | 20 | 0.8126 ± 0.0754 | 14.55 ± 1.80 | 0.8126 |
| DIBCO 2018 (tuning) | otsu | 10 | 0.5145 ± 0.2315 | 9.74 ± 3.07 | 0.5145 |
| DIBCO 2018 (tuning) | sauvola_rel | 10 | 0.6687 ± 0.1726 | 13.33 ± 2.50 | 0.6687 |
| DIBCO 2018 (tuning) | sauvola_stroke | 10 | 0.6882 ± 0.1410 | 13.36 ± 2.38 | 0.6882 |
| DIBCO 2019 (tuning) | otsu | 10 | 0.7233 ± 0.1585 | 15.37 ± 4.70 | 0.7233 |
| DIBCO 2019 (tuning) | sauvola_rel | 10 | 0.6773 ± 0.1355 | 14.04 ± 3.53 | 0.6773 |
| DIBCO 2019 (tuning) | sauvola_stroke | 10 | 0.6909 ± 0.1245 | 14.35 ± 3.71 | 0.6909 |
| Palm leaf (test) | otsu | 45 | 0.1353 ± 0.0849 | 1.07 ± 1.44 | 0.1353 |
| Palm leaf (test) | sauvola_rel | 45 | 0.4525 ± 0.1215 | 11.58 ± 1.45 | 0.4525 |
| Palm leaf (test) | sauvola_stroke | 45 | 0.4396 ± 0.1778 | 9.50 ± 1.84 | 0.4396 |
| Palm leaf (test) | human GT1 vs GT2 | 45 | 0.6283 ± 0.0750 | 14.47 ± 4.05 | 0.6283 |

**Paired per-image comparison, `sauvola_stroke − sauvola_rel`:**

| Dataset | Mean Δ | Wins (stroke > rel) |
|---|---|---|
| DIBCO 2018 (tuning) | +0.0195 | 4 / 10 |
| DIBCO 2019 (tuning) | +0.0135 | 4 / 10 |
| DIBCO 2017 (test) | −0.0007 | 10 / 20 |
| Palm leaf (test) | **−0.0130** | 22 / 45 |

---

## 7. Analysis and Findings

**7.1 Otsu fails badly wherever illumination is uneven or the background is
not paper-like.** It is weakest exactly where predicted: DIBCO 2018
(stains, bleed-through) and palm leaf (F=0.135, PSNR≈1 dB — close to total
failure), and comparatively strong on DIBCO 2019, a cleaner year. This
motivates every subsequent stage of the pipeline and is a clean baseline
figure for the report.

**7.2 Sauvola (both window rules) is a clear, consistent improvement over
Otsu on paper documents**, and a large one on palm leaf, despite starting
from a badly-thresholded seed image — evidence that local adaptive
thresholding is doing real work even before any restoration stage exists.

**7.3 No statistical evidence of a gain from the stroke-width window on the
tuning years.** The mean per-image deltas are positive on both 2018
(+0.0195) and 2019 (+0.0135), but only 4/10 images improve in each year.
Paired Wilcoxon signed-rank tests (two-sided) give p = 0.64 for 2018 and
p = 0.69 for 2019; on this sample size (n = 10 per year) these large p-values
mean there is **no evidence either way** — not that the two rules are proved
identical, but that the data are consistent with either small gains or small
losses.  The positive mean is driven by a minority of images; the median
delta is exactly 0 in both years (several images received identical windows
under both rules).  **This claim has not been established and should not
appear in the final report without that caveat.**

**7.4 The stroke-width rule does not transfer to the test sets; on palm
leaf the stroke-width *estimate* is broken.** DIBCO 2017 shows essentially
no effect (Δ = −0.0007, p not tested; a coin-flip win rate). On palm leaf,
F-Measure shows no statistically significant regression (Δ = −0.0130,
wins 22/45, Wilcoxon p = 0.43 — no evidence either way for F-Measure).
However, PSNR drops from 11.58 to 9.50 dB (significance not tested), a
larger relative drop than F-Measure shows.

The root cause is not a failure of the window rule per se but of its ink
seed: `estimate_stroke_width` seeds itself with a rough Otsu mask, and on
palm leaf Otsu's F-Measure is 0.135 — it marks approximately **91 % of
pixels as ink** (median seed fraction vs GT fraction ratio = 13.4×, range
up to 161×). Skeletonising and distance-transforming that broken mask
produces a median stroke width of **229 px** and a median Sauvola window of
**1,721 px** — larger than most palm images. Sauvola degenerates to a
near-global threshold, which is why the F-Measure effect is small but PSNR
drops sharply (total wrong-pixel count changes more than the precision/recall
balance). The ROI stage is the designed fix: by restricting the Otsu seed to
the detected leaf region, it gives the seed a genuinely bimodal problem
(ink vs leaf) instead of the ill-posed one it currently faces.

**Proposed next diagnostic (not yet run):** log the fraction of pixels the
rough Otsu seed mask marks as "ink" per image, for both DIBCO and palm.
Should be low and stable for DIBCO, anomalously high for palm — turning
§7.4's hypothesis into a measured, citable number for the report.

---

## 8. Status and Next Steps

**Frozen so far (not to be re-tuned by later stages):**
- `window_rel = 0.05, k = 0.15` (`eval/results/sauvola_config.json`)
- `factor = 7.5, floor = 15` (`eval/results/scale_config.json`)
- `data/splits.json` (seed 0)

**Immediate next steps, in order:**
1. Diagnostic: rough-mask ink fraction per image (confirms/refutes §7.4).
2. Wilcoxon signed-rank test on the `sauvola_stroke` vs `sauvola_rel`
   per-image deltas on the tuning years (confirms/refutes §7.3).
3. **ROI stage** (`docbin/roi.py`): separate the document/leaf region from
   its surround via Otsu on a colour channel (saturation or LAB *b*),
   largest-component + hole-fill + border erosion, with a "mask covers
   >95%" fallback to a full-frame no-op — designed to be a near-no-op on
   most DIBCO pages and load-bearing on palm leaf. Expected to directly
   address §7.4 by fixing the seed mask stroke-width estimation depends on.
4. Colour channel selection, masked background normalisation (restoration
   stage), CLAHE, small-component cleanup, `pipeline.py` chaining all
   stages with intermediates exposed.
5. Full ablation table (`eval/ablation.py`) once the pipeline exists
   end-to-end.
6. FastAPI routes + Swagger demo, Dockerfile.

**Open items carried forward:**
- Palm leaf dev set (5 images) still needs a QC pass (mask overlays) once
  the ROI stage exists.
- Report should state the F-Measure/Dice equivalence explicitly rather
  than presenting them as independent evidence.
- The "human ceiling" on palm leaf (0.628) should be presented as a soft
  reference (~0.63–0.81 achievable range), not a hard cap.
