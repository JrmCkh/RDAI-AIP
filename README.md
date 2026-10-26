# RDAI-AIP — YOLO Inference API

A Dockerised [FastAPI](https://fastapi.tiangolo.com/) service that serves an
[Ultralytics YOLO](https://docs.ultralytics.com/) object-detection model (default: `yolo11n.pt`, 80 COCO classes).

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
| GET    | `/model`         | Model path, task, device and class names                 |
| POST   | `/predict`       | Upload an image → JSON list of detections                |
| POST   | `/predict/image` | Upload an image → JPEG with boxes drawn on it            |

Both `/predict` endpoints accept optional `conf` (confidence threshold) and `iou` (NMS threshold) query params.

```bash
curl -F "file=@bus.jpg" "http://localhost:8000/predict?conf=0.4"
curl -F "file=@bus.jpg" http://localhost:8000/predict/image -o annotated.jpg
```

Example response:

```json
{
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
| `MODEL_PATH`    | `/app/models/yolo11n.pt`   | Weights file (any Ultralytics `.pt` model) |
| `DEVICE`        | `cpu`                      | `cpu`, or `0` for the first CUDA GPU       |
| `DEFAULT_CONF`  | `0.25`                     | Default confidence threshold               |
| `DEFAULT_IOU`   | `0.45`                     | Default NMS IoU threshold                  |
| `IMG_SIZE`      | `640`                      | Inference image size                       |
| `MAX_UPLOAD_MB` | `10`                       | Upload size limit                          |

**Custom model:** put your trained weights in `./models/best.pt`, then uncomment the `volumes` and
`MODEL_PATH` lines in `docker-compose.yml`.

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest
```

## Project layout

```
app/
  config.py   # env-var settings
  main.py     # FastAPI app, model loading, endpoints
tests/
  test_api.py
Dockerfile
docker-compose.yml
requirements.txt
```
