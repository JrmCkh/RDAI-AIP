import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from ultralytics import YOLO

from app.config import _discover_models, settings

logger = logging.getLogger("uvicorn.error")

# Tasks whose results carry bounding boxes, which is what /predict returns.
SUPPORTED_TASKS = {"detect", "segment", "pose"}


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


# state["models"]: {name: YOLO}; state["paths"]: {name: weights path};
# state["active"]: name of the model used when a request names none.
state: dict = {}
# Ultralytics models are not guaranteed thread-safe; sync endpoints run in a threadpool.
# One lock per model so requests to different models can run concurrently.
model_locks: dict[str, threading.Lock] = {}
# Guards adding/removing models while the models folder is re-scanned.
registry_lock = threading.Lock()
# path -> (mtime, size) of each loaded file, and of each file that failed to load (so a bad file
# isn't retried on every request; it is retried once it changes, e.g. a tester finishes copying it).
loaded_stamps: dict[str, tuple] = {}
failed_stamps: dict[str, tuple] = {}


def _stamp(path: str) -> tuple:
    try:
        st = Path(path).stat()
        return st.st_mtime, st.st_size
    except OSError:
        return ()


def _load(name: str, path: str) -> bool:
    """Load and warm up one model into the registry. Returns False (and logs) if it can't be used."""
    try:
        model = YOLO(path)
        if model.task not in SUPPORTED_TASKS:
            raise ValueError(f"task {model.task!r} is not supported (need one of {sorted(SUPPORTED_TASKS)})")
        model.to(settings.device)
        # The first predict call initialises the predictor; do it now so no request pays that cost.
        model.predict(
            Image.new("RGB", (settings.img_size, settings.img_size)), device=settings.device, verbose=False
        )
    except Exception as exc:
        # A bad file dropped into the models folder shouldn't take the whole service down,
        # but a model listed explicitly in MODELS is a config error, so fail loudly.
        if not settings.models_discovered:
            raise
        logger.warning("Skipping model %r (%s): %s", name, path, exc)
        failed_stamps[path] = _stamp(path)
        return False
    state["models"][name] = model
    state["paths"][name] = path
    model_locks.setdefault(name, threading.Lock())
    loaded_stamps[path] = _stamp(path)
    failed_stamps.pop(path, None)
    logger.info("Loaded model %r from %s (task=%s, %d classes)", name, path, model.task, len(model.names))
    return True


def _sync_models() -> None:
    """Re-scan the models folder: load new or changed .pt files and drop deleted ones.

    Cheap when nothing changed (one directory listing plus a stat per file), so it runs on every
    request that needs a model. This is what lets testers add models to the mounted folder
    without restarting the container. It does nothing when MODELS is set explicitly.
    """
    if not settings.models_discovered:
        return
    try:
        on_disk = _discover_models(settings.models_dir)
    except ValueError:
        on_disk = {}
    with registry_lock:
        for name in [n for n in state["models"] if n not in on_disk]:
            logger.info("Model %r was removed from %s; unloading it", name, settings.models_dir)
            loaded_stamps.pop(state["paths"].pop(name), None)
            del state["models"][name]
        for name, path in on_disk.items():
            stamp = _stamp(path)
            if loaded_stamps.get(path) == stamp or failed_stamps.get(path) == stamp:
                continue  # unchanged since it was last loaded (or last failed to load)
            if name in state["models"]:
                logger.info("Model file %s changed; reloading %r", path, name)
            _load(name, path)
        if state["active"] not in state["models"]:
            state["active"] = next(iter(state["models"]), None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.update(models={}, paths={}, active=None)
    for name, path in settings.models.items():
        _load(name, path)
    if not state["models"]:
        raise RuntimeError("No models could be loaded; see the warnings above.")
    active = settings.default_model or next(iter(state["models"]))
    if active not in state["models"]:
        raise RuntimeError(f"DEFAULT_MODEL {active!r} failed to load.")
    state["active"] = active
    yield
    state.clear()
    model_locks.clear()
    loaded_stamps.clear()
    failed_stamps.clear()


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
    _sync_models()
    name = name or state["active"]
    if name is None:
        raise HTTPException(status_code=503, detail=f"No models available in {settings.models_dir}.")
    if name not in state["models"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {name!r}. Available: {', '.join(state['models'])}.",
        )
    return name


def _infer(name: str, image: Image.Image, conf: float, iou: float):
    with model_locks[name]:
        model = state["models"].get(name)
        if model is None:  # deleted from the folder between resolving and running
            raise HTTPException(status_code=404, detail=f"Model {name!r} was removed.")
        start = time.perf_counter()
        result = model.predict(
            image, conf=conf, iou=iou, imgsz=settings.img_size, device=settings.device, verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(state.get("models", {}))}


@app.get("/models")
def list_models():
    _sync_models()
    return {
        "active": state["active"],
        "models": [
            {"name": name, "model_path": state["paths"][name], "task": m.task, "num_classes": len(m.names)}
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
        "model_path": state["paths"][name],
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
