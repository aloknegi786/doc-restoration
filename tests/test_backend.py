"""Integration tests for the FastAPI backend (backend/main.py).

If fastapi or httpx is not installed in the environment, these tests are skipped
gracefully rather than failing.
"""
from __future__ import annotations

import io
import zipfile
import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _make_dummy_image_bytes(h: int = 100, w: int = 150) -> bytes:
    """Create a dummy BGR image with paper background and a dark blob."""
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    img[40:60, 50:70] = 30  # dark ink blob
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "pipeline_config" in data
    assert data["pipeline_config"]["window_rel"] == 0.05
    assert data["pipeline_config"]["k"] == 0.15


def test_index_ui_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Historical Document Binarizer" in resp.text


def test_binarize_endpoint():
    img_bytes = _make_dummy_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = client.post("/binarize", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    # Decode returned PNG mask
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.shape == (100, 150)
    # Background should be 255 (white), ink blob should be 0 (black)
    assert (mask[40:60, 50:70] == 0).any()
    assert (mask[:20, :20] == 255).all()


def test_binarize_stages_endpoint():
    img_bytes = _make_dummy_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = client.post("/binarize/stages", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    # Verify ZIP contents
    buf = io.BytesIO(resp.content)
    with zipfile.ZipFile(buf, "r") as zf:
        names = zf.namelist()
        expected = [
            "00_original.png",
            "01_channel.png",
            "02_roi.png",
            "03_normalised.png",
            "04_clahe.png",
            "05_binarized.png",
            "06_cleaned.png",
        ]
        for exp in expected:
            assert exp in names, f"Missing {exp} in stages ZIP"


def test_binarize_invalid_file_rejected():
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    resp = client.post("/binarize", files=files)
    assert resp.status_code == 422
