"""PipelineConfig — single source of truth for every pipeline parameter.

Usage patterns:

1. Load defaults (uses frozen configs from configs/ if present):
       cfg = PipelineConfig.from_frozen_configs()

2. Load from an arbitrary JSON:
       cfg = PipelineConfig.from_json("my_config.json")

3. Ablation — flip one flag and keep everything else:
       cfg_no_roi = dataclasses.replace(cfg, use_roi=False)
       cfg_no_clahe = dataclasses.replace(cfg, use_clahe=False)

4. Save for reproducibility:
       cfg.to_json("eval/results/pipeline_config.json")

All pipeline stages read their parameters exclusively from this object;
no magic defaults are scattered across modules.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

# Default paths to the frozen configs produced by param_sweep.py /
# calibrate_scale.py.  These exist in both eval/results/ and configs/.
_SAUVOLA_CONFIG = Path("configs/sauvola_config.json")
_SCALE_CONFIG   = Path("configs/scale_config.json")


@dataclasses.dataclass
class PipelineConfig:
    # ---- channel selection ------------------------------------------------
    channel: str = "gray"
    """One of 'gray', 'red', 'lab_l', 'pca'.  Colour channel used for
    binarization.  Compared empirically in the ablation."""

    # ---- ROI detection ----------------------------------------------------
    use_roi: bool = True
    """Detect and restrict processing to the document region."""
    roi_margin_frac: float = 0.015
    """Erosion margin as a fraction of image height (removes page border)."""
    roi_close_frac: float = 0.04
    """Morphological closing kernel as a fraction of downsampled height."""
    roi_downsample: int = 4
    """Downsample factor used during ROI detection."""

    # ---- background normalisation -----------------------------------------
    use_normalise: bool = True
    """Apply masked background normalisation (division method)."""
    bg_sigma_frac: float = 0.10
    """Gaussian sigma for background estimation, as a fraction of the
    shorter image side.  Larger → smoother background estimate."""
    bg_downsample: int = 8
    """Downsample factor during background blur (speed optimisation)."""

    # ---- CLAHE -------------------------------------------------------------
    use_clahe: bool = True
    """Apply CLAHE after background normalisation."""
    clahe_grid: tuple[int, int] = (8, 8)
    """Tile grid size (rows, cols) for CLAHE."""
    clahe_clip: float = 2.0
    """Clip limit for CLAHE (relative to average frequency per bin)."""

    # ---- Sauvola binarization (frozen from param_sweep.py) ----------------
    window_rel: float = 0.05
    """Sauvola window size as a fraction of the shorter image side."""
    k: float = 0.15
    """Sauvola sensitivity parameter k."""
    R: float = 128.0
    """Sauvola dynamic range R (do not change; tuned against this value)."""

    # ---- small-component cleanup ------------------------------------------
    use_cleanup: bool = True
    """Remove connected ink components smaller than min_component_size."""
    min_component_size: int = 15
    """Minimum component area in pixels.  Kept low to preserve pencil marks
    and small genuine ink dots present in the ground truth."""

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write this config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        # Convert list → tuple for tuple-typed fields
        if "clahe_grid" in d and isinstance(d["clahe_grid"], list):
            d["clahe_grid"] = tuple(d["clahe_grid"])
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load a config from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_frozen_configs(
        cls,
        sauvola_config: str | Path = _SAUVOLA_CONFIG,
        scale_config: str | Path   = _SCALE_CONFIG,
        **overrides,
    ) -> "PipelineConfig":
        """Build a PipelineConfig from the frozen param-sweep results.

        Reads sauvola_config (window_rel, k) and scale_config (factor, floor)
        — both produced by param_sweep.py / calibrate_scale.py.

        Raises FileNotFoundError if either file is missing; callers must
        ensure the frozen configs exist before calling this.

        Any keyword argument in `overrides` is applied on top of the file
        values, allowing per-experiment tweaks without modifying the files.
        """
        sauvola_p = Path(sauvola_config)
        scale_p   = Path(scale_config)
        if not sauvola_p.exists():
            raise FileNotFoundError(
                f"Frozen Sauvola config not found: {sauvola_p}\n"
                "Run eval/param_sweep.py first."
            )
        if not scale_p.exists():
            raise FileNotFoundError(
                f"Frozen scale config not found: {scale_p}\n"
                "Run eval/calibrate_scale.py first."
            )
        sauvola_cfg = json.loads(sauvola_p.read_text())
        # Note: scale_config (factor/floor) drives stroke-width window, which
        # is not used in the main pipeline — the pipeline uses window_rel.
        # We read it here for completeness but only window_rel and k matter.
        cfg = cls(window_rel=sauvola_cfg["window_rel"], k=sauvola_cfg["k"])
        return dataclasses.replace(cfg, **overrides)

