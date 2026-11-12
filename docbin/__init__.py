"""docbin: Document restoration and binarization library."""
from __future__ import annotations

from docbin.config import PipelineConfig
from docbin.pipeline import binarize, run

__all__ = ["PipelineConfig", "binarize", "run"]
