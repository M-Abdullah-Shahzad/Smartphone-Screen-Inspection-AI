from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.pipeline.run_pipeline import load_pipeline_weights, process_one

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = load_pipeline_weights()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    missing = [str(path) for path in MODEL_PATHS.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing model weights: " + ", ".join(missing))
    from ultralytics import YOLO

    app.state.models = {name: YOLO(str(path)) for name, path in MODEL_PATHS.items()}
    app.state.screen_confidence = 0.5
    yield
    app.state.models.clear()


app = FastAPI(title="Mobile Phone Defect Pipeline", lifespan=lifespan)


@app.get("/", response_class=FileResponse)
async def web_ui() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "src" / "api" / "webui.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": hasattr(app.state, "models") and bool(app.state.models)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), format: str = Query("simple", pattern="^(simple|json)$")) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file")
    temporary_directory = Path(tempfile.mkdtemp(prefix="pipeline_api_"))
    try:
        suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
        input_path = temporary_directory / f"input{suffix}"
        output_dir = temporary_directory / "output"
        input_path.write_bytes(await file.read())
        if cv2.imread(str(input_path)) is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a readable image")
        result = process_one(input_path, output_dir, app.state.models, app.state.screen_confidence, False)
        result["input"] = file.filename or "upload.jpg"
        return result
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)
