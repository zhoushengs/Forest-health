import base64
import io
import json
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from detector import Detector

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
gate = threading.BoundedSemaphore(1)


@asynccontextmanager
async def lifespan(app):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    app.state.detector = Detector(os.environ.get("MODEL_PATH", config["model_path"]))
    yield
    del app.state.detector


app = FastAPI(title="PWD Forest Health Detection", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ready", **app.state.detector.info}


@app.post("/api/detect")
def detect(file: Annotated[UploadFile, File()],
           confidence: Annotated[float, Form(ge=0.01, le=1)] = 0.25,
           iou: Annotated[float, Form(ge=0.01, le=1)] = 0.45):
    if not gate.acquire(blocking=False):
        file.file.close()
        raise HTTPException(429, "Another image is being processed. Please try again shortly.")
    try:
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image size must not exceed 15 MB.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in {"JPEG", "PNG", "WEBP"}:
                        raise HTTPException(415, "Please upload a JPG, PNG or WebP image.")
                    if source.width * source.height > MAX_PIXELS:
                        raise HTTPException(413, "Images must not exceed 25 megapixels.")
                    source.load()
                    image = source.copy()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "Image dimensions are too large.")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, "Unable to read the image. Please check that the file is valid.")
        result, annotated = app.state.detector.predict(image, confidence, iou)
        result.update({"model": app.state.detector.info["model"],
                       "confidence_threshold": confidence, "iou_threshold": iou,
                       "annotated_image": "data:image/jpeg;base64," + base64.b64encode(annotated).decode("ascii")})
        return result
    except HTTPException:
        raise
    except Exception:
        logging.exception("Detection failed")
        raise HTTPException(500, "Detection failed. Please check the server logs.")
    finally:
        file.file.close()
        gate.release()
