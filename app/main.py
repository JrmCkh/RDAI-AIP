import io
import threading
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from ultralytics import YOLO

from app.config import settings


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    filename: str | None
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]


state: dict = {}
# Ultralytics models are not guaranteed thread-safe; sync endpoints run in a threadpool.
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = YOLO(settings.model_path)
    model.to(settings.device)
    # The first predict call initialises the predictor; do it now so no request pays that cost.
    model.predict(Image.new("RGB", (settings.img_size, settings.img_size)), device=settings.device, verbose=False)
    state["model"] = model
    yield
    state.clear()


app = FastAPI(
    title="YOLO Inference API",
    description="Object detection with Ultralytics YOLO, served by FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


def _read_image(file: UploadFile) -> Image.Image:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="File is not a supported image.")


def _infer(image: Image.Image, conf: float, iou: float):
    with model_lock:
        start = time.perf_counter()
        result = state["model"].predict(
            image, conf=conf, iou=iou, imgsz=settings.img_size, device=settings.device, verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model")
def model_info():
    model = state["model"]
    return {
        "model_path": settings.model_path,
        "task": model.task,
        "device": settings.device,
        "num_classes": len(model.names),
        "classes": model.names,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    conf: float = Query(settings.default_conf, ge=0.0, le=1.0),
    iou: float = Query(settings.default_iou, ge=0.0, le=1.0),
):
    image = _read_image(file)
    result, elapsed_ms = _infer(image, conf, iou)

    detections = [
        Detection(
            class_id=int(cls),
            class_name=result.names[int(cls)],
            confidence=round(float(score), 4),
            bbox=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), (round(v, 1) for v in xyxy)))),
        )
        for xyxy, score, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        )
    ]
    return PredictionResponse(
        filename=file.filename,
        image_width=image.width,
        image_height=image.height,
        inference_ms=round(elapsed_ms, 2),
        detections=detections,
    )


@app.post("/predict/image", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(settings.default_conf, ge=0.0, le=1.0),
    iou: float = Query(settings.default_iou, ge=0.0, le=1.0),
):
    image = _read_image(file)
    result, _ = _infer(image, conf, iou)
    annotated_bgr = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
