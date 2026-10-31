# RDAI-AIP â€” YOLO Inference API

A Dockerised [FastAPI](https://fastapi.tiangolo.com/) service that serves an
[Ultralytics YOLO](https://docs.ultralytics.com/) object-detection models. Two models are loaded by default —
`yolo11n` (nano, fastest) and `yolo11s` (small, more accurate), both 80 COCO classes — and you can pick one per
request or switch the default at runtime.

## Quick start (Docker)

```bash
docker compose up --build
```

Then open the interactive docs at <http://localhost:8000/docs>.

Without compose:

```bash
docker build -t yolo-fastapi .
docker run --rm -p 8000:8000 yolo-fastapi
```

The model weights are downloaded at build time and baked into the image, so the container runs offline.

## Endpoints

| Method | Path             | Description                                              |
|--------|------------------|----------------------------------------------------------|
| GET    | `/health`        | Liveness check                                           |
| GET    | `/models`        | List loaded models and which one is active               |
| GET    | `/model`         | Path, task, device and class names (`?model=` optional)  |
| PUT    | `/model`         | Switch the active model: `{"name": "yolo11s"}`           |
| POST   | `/predict`       | Upload an image â†’ JSON list of detections                |
| POST   | `/predict/image` | Upload an image â†’ JPEG with boxes drawn on it            |

Both `/predict` endpoints accept optional `conf` (confidence threshold), `iou` (NMS threshold) and `model`
(model name; defaults to the active model) query params. The JSON response includes a `model` field and the
image response an `X-Model` header, so you can see which model answered.

```bash
curl -F "file=@bus.jpg" "http://localhost:8000/predict?conf=0.4"
curl -F "file=@bus.jpg" "http://localhost:8000/predict?model=yolo11s"
curl -F "file=@bus.jpg" http://localhost:8000/predict/image -o annotated.jpg

# Change the default model for all subsequent requests
curl -X PUT -H "Content-Type: application/json" -d '{"name": "yolo11s"}' http://localhost:8000/model
```

Example response:

```json
{
  "model": "yolo11n",
  "filename": "bus.jpg",
  "image_width": 810,
  "image_height": 1080,
  "inference_ms": 61.3,
  "detections": [
    {"class_id": 5, "class_name": "bus", "confidence": 0.9402,
     "bbox": {"x1": 3.8, "y1": 229.4, "x2": 796.2, "y2": 728.4}}
  ]
}
```

## Configuration

Set via environment variables (see `docker-compose.yml`):

| Variable        | Default                    | Meaning                                    |
|-----------------|----------------------------|--------------------------------------------|
| `MODELS`        | `yolo11n=models/yolo11n.pt,yolo11s=models/yolo11s.pt` (Docker: `/app/models/…`) | Comma-separated `name=path` list of models to load (a bare path is named after its file stem) |
| `DEFAULT_MODEL` | first entry in `MODELS`    | Model used when a request doesn't name one |
| `DEVICE`        | `cpu`                      | `cpu`, or `0` for the first CUDA GPU       |
| `DEFAULT_CONF`  | `0.25`                     | Default confidence threshold               |
| `DEFAULT_IOU`   | `0.45`                     | Default NMS IoU threshold                  |
| `IMG_SIZE`      | `640`                      | Inference image size                       |
| `MAX_UPLOAD_MB` | `10`                       | Upload size limit                          |

**Custom model:** put your trained weights in `./models/best.pt`, then uncomment the `volumes` and
`MODELS` lines in `docker-compose.yml`. It's then available as `?model=custom`.

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
uvicorn app.main:app --reload   # loads the bundled weights in models/
pytest
```

## Project layout

```
app/
  config.py   # env-var settings (MODELS / DEFAULT_MODEL parsing)
  main.py     # FastAPI app, multi-model loading, endpoints
tests/
  test_api.py
models/
  yolo11n.pt  # bundled weights so the models work straight after cloning
  yolo11s.pt
Dockerfile
docker-compose.yml
requirements.txt
```
