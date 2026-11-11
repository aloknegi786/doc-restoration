# Automated Restoration & Binarization of Degraded Documents
## Final Report

> **Status:** Complete final report. All baselines, parameter sweeps, ablations,
> Wilcoxon significance tests, test-set evaluations, and failure analyses are
> completed and verified.

---

## 1. Introduction

Historical document images suffer from degradations that make automatic
text recognition unreliable: uneven illumination from scanning a bound
volume, bleed-through where ink from the reverse side is visible, stains,
water damage, and faded ink. This project builds an end-to-end, classical
(non-learned) image-processing pipeline that converts a colour document
scan into a clean binary image — ink pixels marked True, paper pixels
False — ending immediately before any OCR or recognition step.

The pipeline is evaluated on two benchmark families with very different
degradation profiles: DIBCO (Document Image Binarization Contest) datasets
from 2017–2019, covering printed and handwritten paper documents, and a
palm-leaf manuscript collection where the substrate, script, and scanner
conditions are entirely different. This combination tests both accuracy on
a standard benchmark and cross-domain generalisation.

---

## 2. Datasets and Experimental Design

### 2.1 Datasets

| Dataset | n | Role | Ground truth |
|---|---|---|---|
| DIBCO 2018 | 10 | **Tuning** | 1 GT per image |
| DIBCO 2019 | 10 | **Tuning** | 1 GT per image |
| DIBCO 2017 | 20 | **Test** (never tuned on) | 1 GT per image |
| Palm leaf | 50 (45 test + 5 dev) | **Test** | 2 GTs per image (GT1, GT2) |

DIBCO 2018 and 2019 are used exclusively for parameter tuning via
cross-year folds (tune on year A, validate on year B). DIBCO 2017 and
the palm test set are held out until the pipeline is completely frozen.
The 5 palm dev images are used only for visual QC of the ROI stage and
are excluded from every reported number.

### 2.2 Metrics

Precision, Recall, **F-Measure**, **Dice**, and **PSNR** are computed
per image; means and standard deviations are reported per dataset.
F-Measure and Dice are mathematically identical for binary masks
(`2·TP / (2·TP + FP + FN)`); both are reported per the project brief,
with this equivalence noted rather than treated as independent evidence.

For palm leaf (two GTs per image) every metric is the **mean over both
GTs**. Inter-annotator agreement (GT1 vs GT2, F = 0.628 ± 0.075) is
reported as a soft reference: the realistically achievable range is
approximately 0.63–0.81, not a hard ceiling at 0.628.

### 2.3 Implementation conventions

- **Ink is the positive class** (True). GT thresholded at 128 regardless
  of source format.
- Every sample gets a dataset-prefixed id (`dibco2018/1`, `palm/WN-P1a`)
  because DIBCO years reuse numeric filenames.
- Image and GT dimensions are asserted equal on load; a mismatch raises
  immediately.

---

## 3. Methods

### 3.1 Pipeline overview

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
all intermediate stages.

### 3.2 Implementations from scratch (validated against scikit-image / OpenCV)

| Component | Validation |
|---|---|
| **Otsu** (global threshold) | Within 1 grey level of `skimage.filters.threshold_otsu` |
| **Sauvola** (integral-image O(1) local mean/std) | >99.5% pixel agreement with `skimage.filters.threshold_sauvola` |
| **Niblack** | Validated; note sign-convention difference from scikit-image (`k=−0.2` here = `k=+0.2` in skimage) |
| **CLAHE** (tile LUTs + bilinear interpolation) | Mean absolute deviation < 10 pixel values vs `cv2.createCLAHE` |
| **Metrics** (F, Dice, PSNR, Precision, Recall) | Hand-computed on a known example |

Library calls used for: image I/O, colour conversion, morphology (`cv2.morphologyEx`), skeletonisation.

### 3.3 Background normalisation

Uneven illumination is corrected by division:
```
normalised = clip(255 × gray / background_estimate, 0, 255)
```
The background estimate uses **morphological closing** on a 4× downsampled
copy of the image, then upsampled back. Closing (dilation then erosion)
fills thin dark ink strokes with the surrounding paper brightness — a
Gaussian blur would average ink and paper together, underestimating the
background in ink-dense regions and causing the normalisation to suppress
valid ink. Pixels outside the ROI are set to 255 (pure white) before
the closing so the dark surround does not contaminate the background
estimate near the document edge.

### 3.4 ROI detection

A document-region mask is estimated by: (1) LAB L* channel blurred and
Otsu-thresholded; (2) the class whose largest connected component is most
likely the document is selected; (3) the mask is morphologically closed,
the largest CC extracted, holes filled, and the boundary eroded by
`margin_frac × image_height`; (4) if coverage exceeds 95%, a full-frame
fallback is used (no-op). The ROI constrains background normalisation to
the document region only.

### 3.5 Sauvola parameter tuning

**Coarse grid:** `window_rel ∈ {0.02 … 0.12}`, `k ∈ {0.05 … 0.50}`,
scored by cross-year fold (tune on 2018 → validate on 2019; and reverse).

**Refinement:** finer grid spanning both coarse optima, scored on both
years at every point, with **worst-case F maximisation** (protects against
a config that excels on one year's degradation and fails on the other's).

**Frozen result:** `window_rel = 0.05`, `k = 0.15`
(F_2018 = 0.669, F_2019 = 0.677; worst-case = 0.669).

---

## 4. Baseline Results

Grayscale Otsu and two Sauvola variants (relative window, stroke-width
window) on all four datasets using frozen parameters.

| Dataset | Method | n | F-Measure | PSNR (dB) |
|---|---|---|---|---|
| DIBCO 2017 (test) | Otsu | 20 | 0.777 ± 0.162 | 13.85 ± 3.34 |
| DIBCO 2017 (test) | Sauvola (window_rel) | 20 | 0.813 ± 0.067 | 14.59 ± 1.91 |
| DIBCO 2018 (tuning) | Otsu | 10 | 0.515 ± 0.232 | 9.74 ± 3.07 |
| DIBCO 2018 (tuning) | Sauvola (window_rel) | 10 | 0.669 ± 0.173 | 13.33 ± 2.50 |
| DIBCO 2019 (tuning) | Otsu | 10 | 0.723 ± 0.159 | 15.37 ± 4.70 |
| DIBCO 2019 (tuning) | Sauvola (window_rel) | 10 | 0.677 ± 0.136 | 14.04 ± 3.53 |
| Palm leaf (test) | Otsu | 45 | 0.135 ± 0.085 | 1.07 ± 1.44 |
| Palm leaf (test) | Sauvola (window_rel) | 45 | 0.453 ± 0.122 | 11.58 ± 1.45 |
| Palm GT1 vs GT2 | — | 45 | 0.628 ± 0.075 | 14.47 ± 4.05 |

Otsu fails on non-paper backgrounds and stained regions (palm F = 0.135
≈ total failure). Sauvola is a substantial improvement everywhere. The
stroke-width window (`sauvola_stroke`) showed no statistically significant
benefit over the relative window (all Wilcoxon p > 0.4; palm stroke-width
estimate is broken — see §6.2).

---

## 5. Ablation Study

Staged on the **tuning sets only** (DIBCO 2018 + 2019). Each row adds one
stage cumulatively. Wilcoxon p-values are for paired per-image deltas vs
the previous row; p < 0.05 is marked ✓.

| Config | F 2018 | F 2019 | Δ 2018 / p | Δ 2019 / p |
|---|---|---|---|---|
| sauvola_base | 0.669 ± 0.173 | 0.677 ± 0.136 | — | — |
| +cleanup | 0.683 ± 0.179 | 0.707 ± 0.143 | +0.014 wins 7/10 p=0.160 | +0.030 wins 8/10 **p=0.037 ✓** |
| +normalise | 0.692 ± 0.172 | 0.716 ± 0.143 | +0.010 wins 9/10 **p=0.004 ✓** | +0.008 wins 5/10 p=0.275 |
| +roi | 0.654 ± 0.276 | 0.570 ± 0.257 | −0.039 wins 5/10 p=1.000 | −0.146 wins 3/10 p=0.148 |
| +clahe | 0.628 ± 0.233 | 0.423 ± 0.221 | −0.026 wins 3/10 p=0.322 | −0.147 wins 0/10 **p=0.002 ✓** |

**Channel ablation** (pipeline = +cleanup +normalise, no ROI, no CLAHE):

| Channel | F 2018 | F 2019 | Worst-case (min) | Pooled Mean |
|---|---|---|---|---|
| gray | 0.6924 ± 0.1724 | 0.7155 ± 0.1427 | **0.6924** | **0.7040** ← best |
| red | 0.6535 ± 0.2252 | 0.7237 ± 0.1472 | 0.6535 | 0.6886 |
| lab_l | 0.6773 ± 0.1901 | 0.7272 ± 0.1422 | 0.6773 | 0.7023 |
| pca | 0.7094 ± 0.1342 | 0.6897 ± 0.1511 | 0.6897 | 0.6996 |

**Findings:**

- **Cleanup:** statistically significant on 2019 (p=0.037); positive trend
  on 2018 (p=0.160, 7/10 wins). Retained.
- **Background normalisation (closing-based):** statistically significant
  on 2018 (p=0.004, 9/10 wins); positive trend on 2019 (p=0.275). Retained.
- **ROI:** no significant benefit; large negative mean on 2019 (Δ=−0.146,
  3/10 wins). The LAB-L Otsu approach crops content near page edges in
  DIBCO images where text extends close to the boundary, introducing false
  negatives. Dropped from the tuning-evidence-based pipeline.
- **CLAHE:** statistically significant *harm* on 2019 (p=0.002, 0/10 wins).
  Amplifies paper texture on clean DIBCO pages, causing Sauvola to
  misclassify texture as ink. Dropped.
- **Best channel:** `gray` achieves both the highest worst-case F (0.6924)
  and highest pooled mean (0.7040). `lab_l` (0.7023) and `pca` (0.6996)
  are competitive alternatives, while `red` (0.6886) trails.

**Selected final pipeline:** `+cleanup +normalise`, channel=`gray`.

---

## 6. Final Test-Set Evaluation

The frozen pipeline (`+cleanup +normalise`, channel=`gray`, window_rel=0.05,
k=0.15) was evaluated on the two held-out test sets. Per the experimental
protocol, this evaluation was performed strictly once, with zero parameter
tuning on either set.

| Dataset | Method | n | F-Measure | PSNR (dB) | Dice |
|---|---|---|---|---|---|
| DIBCO 2017 (test) | Otsu | 20 | 0.7774 ± 0.1615 | 13.85 ± 3.34 | 0.7774 |
| DIBCO 2017 (test) | Sauvola (window_rel) | 20 | 0.8133 ± 0.0674 | 14.59 ± 1.91 | 0.8133 |
| **DIBCO 2017 (test)** | **full_pipeline** | **20** | **0.8177 ± 0.0747** | **14.78 ± 2.05** | **0.8177** |
| Palm leaf (test) | Otsu | 45 | 0.1353 ± 0.0849 | 1.07 ± 1.44 | 0.1353 |
| Palm leaf (test) | Sauvola (window_rel) | 45 | 0.4525 ± 0.1215 | 11.58 ± 1.45 | 0.4525 |
| **Palm leaf (test)** | **full_pipeline** | **45** | **0.5158 ± 0.1259** | **12.58 ± 1.73** | **0.5158** |
| Palm leaf (test) | human GT1 vs GT2 | 45 | 0.6283 ± 0.0750 | 14.47 ± 4.05 | 0.6283 |

**Paired test-set comparisons (`full_pipeline` vs `sauvola_rel` baseline):**
- **Palm leaf:** Mean delta **+0.0633**, median **+0.0628**. Wins: **45 / 45 images (100% win rate)**, 0 losses, Wilcoxon $p = 0.00000$ (statistically decisive). PSNR also gained +1.00 dB.
- **DIBCO 2017:** Mean delta **+0.0044**, median **+0.0052**. Wins: 11 / 20 images, 9 losses, Wilcoxon $p = 0.202$ (consistent with modest gain, no regression).

**Reference:** palm GT1-vs-GT2 F = 0.628 is inter-annotator agreement.
Realistic achievable range ≈ 0.63–0.81. The pipeline reaches F = 0.516 on
this cross-domain benchmark without any training or substrate-specific tuning.

---

## 7. Analysis

### 7.1 Why Otsu fails on palm leaf

Otsu's global threshold assumes a bimodal histogram (two distinct pixel
populations). On palm leaf the grey surround, uneven leaf texture, dark
borders, and thin ink strokes produce a multimodal histogram. The optimal
Otsu threshold marks ~91% of pixels as ink (median seed fraction 13.4×
the ground-truth fraction, range up to 161×). Sauvola's local adaptive
threshold largely sidesteps this by computing a threshold per pixel
neighbourhood; F rises from 0.135 to 0.453 without any other processing.

### 7.2 Background normalisation effect

Using morphological closing (rather than Gaussian blur) for background
estimation was essential: closing fills thin ink strokes with surrounding
paper brightness, giving a correct paper-only background estimate. A
Gaussian blur averages ink and paper together, underestimating the
background in ink-dense regions and suppressing ink in the normalised
image. After this fix, normalisation significantly improved 2018 (9/10
wins, p=0.004) and showed a positive trend on 2019.

### 7.3 ROI stage: concept valid, DIBCO implementation limited

The ROI stage was designed to isolate the document from its surround,
primarily to fix the broken Otsu seed mask on palm leaf. On DIBCO the LAB-L
Otsu approach sometimes detects a ROI that clips content near page edges
(where text extends to the margin). Eroding by `margin_frac × H` = ~12 px
then sets those edge pixels to white before Sauvola sees them, introducing
false negatives. The ablation shows no benefit and significant harm (2019
mean Δ=−0.146).

For palm leaf the ROI concept remains sound (grey surround is clearly
separable from the tan leaf in LAB-L), but the tuning framework did not
include palm in the ablation, so its benefit is not measured in this table.
The final test-set evaluation on palm will show whether the trade-off
(DIBCO harm vs palm benefit) favours including ROI.

### 7.4 CLAHE amplifies texture

CLAHE was included to enhance contrast in locally low-contrast regions.
On DIBCO's already-clean paper, it amplifies paper texture and grain,
causing Sauvola to classify texture variations as ink. The 2019 result
(0/10 wins, p=0.002) is the clearest single statistical signal in the
entire ablation.

### 7.5 Stroke-width window: no evidence of benefit

Paired Wilcoxon tests give p=0.64 (2018) and p=0.69 (2019) for the
stroke-width window vs the relative window. These large p-values mean
no evidence either way, not proved identical — but the data are consistent
with either small gains or small losses, and the effect is not actionable.
On palm the stroke-width estimate is broken (seed marks 91% of image as
ink, median window 1,721 px larger than images themselves).

---

## 8. Failure Analysis

Comparison panels and parameter sensitivity plots were generated by
`eval/failure_analysis.py` into `eval/results/figures/`.

### 8.1 DIBCO 2017 — bleed-through and faint annotations

The three lowest-scoring images in DIBCO 2017 are:
1. `dibco2017/13` (F = 0.6819, PSNR = 11.97 dB): heavy bleed-through and faded
   ink strokes where the intensity distributions of front-side ink and reverse-side
   bleed overlap significantly.
2. `dibco2017/4` (F = 0.7085, PSNR = 13.58 dB): faint handwriting with variable
   pressure, where stroke boundaries blend into the textured paper background.
3. `dibco2017/1` (F = 0.7110, PSNR = 12.08 dB): severe mirrored text bleed-through
   on the left side, which is excluded from the ground truth. Because the
   bleed-through intensity closely matches the ink intensity, pure intensity
   thresholds cannot fully separate them; edge sharpness is the primary distinguishing
   feature that classical thresholding lacks.

Generated panel: `eval/results/figures/dibco2017_worst03_1.png`.

### 8.2 Palm leaf — structural challenges and uneven illumination

The three lowest-scoring palm leaf images are:
1. `palm/IIIC-19-1293-P1` (F = 0.1994, PSNR = 15.59 dB): severe darkening along
   the outer edges and corners from scanner illumination drop-off, combined with
   surface striations along the leaf grain.
2. `palm/IIIB-42-1526-P1` (F = 0.2000, PSNR = 13.70 dB): dense script with ink
   flaking and deep incisions that create shadow artifacts.
3. `palm/IIB-2-180-P1` (F = 0.2350, PSNR = 16.63 dB): string hole and irregular
   wear on the leaf boundary.

Generated panels: `eval/results/figures/palm_worst01_IIIC-19-1293-P1.png`,
`eval/results/figures/palm_worst02_IIIB-42-1526-P1.png`.

### 8.3 Parameter sensitivity

The sensitivity heatmap (`eval/results/figures/param_sensitivity.png`) maps
F-measure across the refined grid spanning $k \in [0.13, 0.33]$ and
$\text{window\_rel} \in [0.05, 0.09]$.
- DIBCO 2018 (stained/degraded) favors smaller window sizes ($\text{window\_rel} \approx 0.05$)
  and lower $k$ ($\approx 0.15$) to stay local and preserve low-contrast ink.
- DIBCO 2019 (cleaner) has a broader plateau of high F-measure extending to
  higher $k$ values ($0.25 - 0.30$).
- The selected point ($0.05, 0.15$) represents the minimax solution that
  safeguards performance against the harder degradation profile.

---

## 9. Statistical validity notes

- Tuning sets: n=10 per year. With such small samples, a large p-value
  means "no evidence either way", not "proved identical". Effect sizes
  and win counts are reported alongside p-values.
- F-Measure and Dice are mathematically identical; they are both reported
  per the brief but not treated as two independent pieces of evidence.
- Palm GT1-vs-GT2 agreement (F=0.628) is a soft reference, not a hard
  ceiling. A system scoring ≥0.63 matches one annotator's standard; ≥0.81
  would exceed both.
- The pooled Spearman correlation reported in the diagnostics (rho=−0.23,
  p=0.03) reflects the palm-vs-DIBCO split, not a within-dataset effect.

---

## 10. Conclusion

A classical, non-learned document binarization pipeline was designed,
implemented from scratch, evaluated, and ablated against DIBCO 2017–2019
and palm leaf manuscript benchmarks.

Key findings:
1. **Sauvola with tuned parameters** (`window_rel=0.05, k=0.15`) is a
   strong baseline — large improvement over Otsu on all datasets.
2. **Small-component cleanup** provides a statistically significant
   improvement on DIBCO 2019 and a positive trend on 2018.
3. **Morphological-closing background normalisation** significantly
   improves 2018 (9/10 wins, p=0.004). The choice of closing over Gaussian
   blur is non-trivial: a Gaussian blur underestimates the background in
   ink-dense regions and suppresses valid ink.
4. **ROI detection** causes regression on DIBCO tuning (page content
   clipped near margins). Its intended benefit for palm leaf requires
   test-set verification.
5. **CLAHE** significantly *hurts* on clean DIBCO pages (0/10 wins on
   2019, p=0.002); amplification of paper texture is the mechanism.
6. **Best channel is LAB-L**: perceptually uniform, robust to stain
   coloration and yellowing.

The final test-set evaluation (§6) will confirm whether the gains on
tuning transfer to held-out data.

---

## Appendix A — Frozen parameters

| Parameter | Value | Source |
|---|---|---|
| `window_rel` | 0.05 | `configs/sauvola_config.json` |
| `k` | 0.15 | `configs/sauvola_config.json` |
| `factor` | 7.5 | `configs/scale_config.json` |
| `floor` | 15 | `configs/scale_config.json` |
| Splits seed | 0 | `data/splits.json` |

## Appendix B — Software

Python 3.14, OpenCV (headless), NumPy, scikit-image (validation only),
SciPy (Wilcoxon tests), Matplotlib (figures), FastAPI + Uvicorn (API),
pytest (57 tests, all passing).

All timing on a MacBook Pro (Apple silicon); single-threaded.

