import io
import threading
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
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
    model: str
    filename: str | None
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]


class ActiveModelRequest(BaseModel):
    name: str = Field(..., description="Name of a loaded model (see GET /models).")


# state["models"]: {name: YOLO}; state["active"]: name of the model used when a request names none.
state: dict = {}
# Ultralytics models are not guaranteed thread-safe; sync endpoints run in a threadpool.
# One lock per model so requests to different models can run concurrently.
model_locks: dict[str, threading.Lock] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup = Image.new("RGB", (settings.img_size, settings.img_size))
    models = {}
    for name, path in settings.models.items():
        model = YOLO(path)
        model.to(settings.device)
        # The first predict call initialises the predictor; do it now so no request pays that cost.
        model.predict(warmup, device=settings.device, verbose=False)
        models[name] = model
        model_locks[name] = threading.Lock()
    state["models"] = models
    state["active"] = settings.default_model
    yield
    state.clear()
    model_locks.clear()


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


def _resolve_model(name: str | None) -> str:
    name = name or state["active"]
    if name not in state["models"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {name!r}. Available: {', '.join(state['models'])}.",
        )
    return name


def _infer(name: str, image: Image.Image, conf: float, iou: float):
    with model_locks[name]:
        start = time.perf_counter()
        result = state["models"][name].predict(
            image, conf=conf, iou=iou, imgsz=settings.img_size, device=settings.device, verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(state.get("models", {}))}


@app.get("/models")
def list_models():
    return {
        "active": state["active"],
        "models": [
            {"name": name, "model_path": settings.models[name], "task": m.task, "num_classes": len(m.names)}
            for name, m in state["models"].items()
        ],
    }


@app.get("/model")
def model_info(model: str | None = Query(None, description="Model name; defaults to the active model.")):
    name = _resolve_model(model)
    m = state["models"][name]
    return {
        "name": name,
        "active": name == state["active"],
        "model_path": settings.models[name],
        "task": m.task,
        "device": settings.device,
        "num_classes": len(m.names),
        "classes": m.names,
    }


@app.put("/model")
def set_active_model(body: ActiveModelRequest):
    """Switch the model used by /predict endpoints when no `model` query param is given."""
    state["active"] = _resolve_model(body.name)
    return {"active": state["active"]}


MODEL_QUERY = Query(None, description="Model name to use; defaults to the active model (see GET /models).")


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    conf: float = Query(settings.default_conf, ge=0.0, le=1.0),
    iou: float = Query(settings.default_iou, ge=0.0, le=1.0),
    model: str | None = MODEL_QUERY,
):
    name = _resolve_model(model)
    image = _read_image(file)
    result, elapsed_ms = _infer(name, image, conf, iou)

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
        model=name,
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
    model: str | None = MODEL_QUERY,
):
    name = _resolve_model(model)
    image = _read_image(file)
    result, _ = _infer(name, image, conf, iou)
    annotated_bgr = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return Response(content=buffer.tobytes(), media_type="image/jpeg", headers={"X-Model": name})
