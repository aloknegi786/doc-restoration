from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class PipelineConfig:
    channel: str = "gray"
    use_roi: bool = True
    roi_margin_frac: float = 0.015
    roi_close_frac: float = 0.04
    roi_downsample: int = 4
    use_normalise: bool = True
    bg_sigma_frac: float = 0.10
    bg_downsample: int = 4
    use_clahe: bool = True
    clahe_grid: tuple[int, int] = (8, 8)
    clahe_clip: float = 2.0
    window_rel: float = 0.05
    k: float = 0.15
    R: float = 128.0
    use_cleanup: bool = True
    min_component_size: int = 15

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        if "clahe_grid" in d and isinstance(d["clahe_grid"], list):
            d["clahe_grid"] = tuple(d["clahe_grid"])
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))
