"""FastAPI backend — document binarization service.

Three routes:

  GET  /health
       Returns {"status": "ok"} and the pipeline config currently loaded.

  POST /binarize
       Accepts a multipart image file upload.
       Returns the binary mask as a PNG (ink = black, paper = white).

  POST /binarize/stages
       Accepts a multipart image file upload.
       Returns a ZIP archive containing one PNG per pipeline stage:
         channel.png, roi.png, normalised.png, clahe.png,
         binarized.png, cleaned.png (the final output).

Demo: run the server and visit http://localhost:8000/docs for Swagger UI.

Usage:
    uvicorn backend.main:app --reload --port 8000

Configuration: on startup the app loads configuration from
configs/pipeline_config.json. Any stage field in PipelineConfig can be overridden
via query parameters on each endpoint (e.g. ?channel=lab_l or ?use_clahe=false).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
except ImportError as e:
    raise ImportError(
        "FastAPI is not installed. Run: pip install fastapi uvicorn[standard]"
    ) from e

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbin.config import PipelineConfig
from docbin.pipeline import run as pipeline_run

_PIPELINE_CONFIG = Path("configs/pipeline_config.json")


def _load_default_config() -> PipelineConfig:
    if _PIPELINE_CONFIG.exists():
        return PipelineConfig.from_json(_PIPELINE_CONFIG)
    return PipelineConfig()


_DEFAULT_CFG = _load_default_config()

app = FastAPI(
    title="Document Binarization API",
    description=(
        "Classical image-processing pipeline that converts degraded historical "
        "document images into clean binary (ink / paper) images. "
        "See `/binarize/stages` to inspect every intermediate step."
    ),
    version="1.0.0",
)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. Upload a valid JPEG, PNG, BMP or TIFF file.",
        )
    return bgr


def _bool_mask_to_png(mask: np.ndarray) -> bytes:
    """Ink = black (0), paper = white (255)."""
    img = np.where(mask, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _gray_to_png(gray: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", gray)
    return buf.tobytes()


def _roi_overlay_png(bgr: np.ndarray, roi: np.ndarray) -> bytes:
    """Semi-transparent green overlay of the ROI mask."""
    from docbin.roi import draw_roi_overlay
    vis = draw_roi_overlay(bgr, roi)
    _, buf = cv2.imencode(".png", vis)
    return buf.tobytes()


def _build_cfg(
    channel: str,
    use_roi: bool,
    use_normalise: bool,
    use_clahe: bool,
    use_cleanup: bool,
) -> PipelineConfig:
    import dataclasses
    return dataclasses.replace(
        _DEFAULT_CFG,
        channel=channel,
        use_roi=use_roi,
        use_normalise=use_normalise,
        use_clahe=use_clahe,
        use_cleanup=use_cleanup,
    )


_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_SAMPLE_DIR   = Path(__file__).resolve().parent.parent / "sample_images"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Interactive demo interface."""
    if _STATIC_INDEX.exists():
        return HTMLResponse(content=_STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Static UI not found</h3>", status_code=404)


@app.get("/samples/{filename}", summary="Get sample historical document image")
def get_sample_image(filename: str):
    """Retrieve sample historical document images for instant testing without uploading."""
    target = (_SAMPLE_DIR / filename).resolve()
    if target.is_file() and str(target).startswith(str(_SAMPLE_DIR.resolve())):
        return FileResponse(str(target))
    raise HTTPException(status_code=404, detail="Sample image not found")


@app.get("/health", summary="Health check")
def health():
    """Return service status and the active pipeline configuration."""
    return {
        "status": "ok",
        "pipeline_config": _DEFAULT_CFG.to_dict(),
    }


@app.post(
    "/binarize",
    summary="Binarize a document image",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def binarize(
    file: Annotated[UploadFile, File(description="Document image (JPEG, PNG, BMP, TIFF)")],
    channel: Annotated[str,  Query(description="Channel: gray | red | lab_l | pca")] = _DEFAULT_CFG.channel,
    use_roi: Annotated[bool, Query(description="Apply ROI detection")] = _DEFAULT_CFG.use_roi,
    use_normalise: Annotated[bool, Query(description="Apply background normalisation")] = _DEFAULT_CFG.use_normalise,
    use_clahe: Annotated[bool, Query(description="Apply CLAHE")] = _DEFAULT_CFG.use_clahe,
    use_cleanup: Annotated[bool, Query(description="Remove small ink components")] = _DEFAULT_CFG.use_cleanup,
):
    """
    Upload a document image and receive the binary ink mask as a PNG.

    - **Ink** is rendered as **black** (0).
    - **Paper / background** is rendered as **white** (255).
    - Use query parameters to toggle individual pipeline stages for quick experimentation.
    """
    data = await file.read()
    bgr = _decode_image(data)
    cfg = _build_cfg(channel, use_roi, use_normalise, use_clahe, use_cleanup)
    stages = pipeline_run(bgr, cfg)
    png_bytes = _bool_mask_to_png(stages["cleaned"])
    return Response(content=png_bytes, media_type="image/png")


@app.post(
    "/binarize/stages",
    summary="Binarize and return all intermediate stages as a ZIP",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/zip": {}}}},
)
async def binarize_stages(
    file: Annotated[UploadFile, File(description="Document image (JPEG, PNG, BMP, TIFF)")],
    channel: Annotated[str,  Query()] = _DEFAULT_CFG.channel,
    use_roi: Annotated[bool, Query()] = _DEFAULT_CFG.use_roi,
    use_normalise: Annotated[bool, Query()] = _DEFAULT_CFG.use_normalise,
    use_clahe: Annotated[bool, Query()] = _DEFAULT_CFG.use_clahe,
    use_cleanup: Annotated[bool, Query()] = _DEFAULT_CFG.use_cleanup,
):
    """
    Upload a document image and receive a ZIP archive with one PNG per pipeline stage.

    Archive contents:
    - `00_original.png` — the uploaded image (colour)
    - `01_channel.png`  — after channel selection (grayscale)
    - `02_roi.png`      — ROI mask overlaid on the original (green fill)
    - `03_normalised.png` — after background normalisation
    - `04_clahe.png`    — after CLAHE contrast enhancement
    - `05_binarized.png` — after Sauvola thresholding (before cleanup)
    - `06_cleaned.png`  — final binary output (after small-component removal)
    """
    data = await file.read()
    bgr = _decode_image(data)
    cfg = _build_cfg(channel, use_roi, use_normalise, use_clahe, use_cleanup)
    stages = pipeline_run(bgr, cfg)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _, orig_buf = cv2.imencode(".png", bgr)
        zf.writestr("00_original.png", orig_buf.tobytes())

        zf.writestr("01_channel.png", _gray_to_png(stages["channel"]))

        zf.writestr("02_roi.png", _roi_overlay_png(bgr, stages["roi"]))

        zf.writestr("03_normalised.png", _gray_to_png(stages["normalised"]))

        zf.writestr("04_clahe.png", _gray_to_png(stages["clahe"]))

        zf.writestr("05_binarized.png", _bool_mask_to_png(stages["binarized"]))

        zf.writestr("06_cleaned.png", _bool_mask_to_png(stages["cleaned"]))

    buf.seek(0)
    stem = Path(file.filename or "output").stem
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_stages.zip"'},
    )

