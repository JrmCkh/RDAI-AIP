# RDAI-AIP — Dockerised FastAPI serving of a model

A Dockerised FastAPI service that serves YOLO object-detection models. Two models are loaded by default `yolo11n` and `yolo11s`, both with 80 COCO classes.

## Quick start (Docker)

```bash
docker compose up --build
```

Then open the interactive docs at <http://localhost:8000/docs> to test out the endpoints.

## Test images

The `test_imgs/` folder has sample images to upload to the `/predict` endpoints:

| Image           | What's in it                   |
|-----------------|--------------------------------|
| `bus.jpg`       | A bus and people on a street   |
| `people.jpg`    | Two people in suits, with ties |
| `cats.jpg`      | Two cats and a remote on a couch |
| `bear.jpg`      | A bear                         |
| `stop_sign.jpg` | A stop sign and a truck        |

Try the same image with `?model=yolo11n` and `?model=yolo11s`. On `people.jpg` and `stop_sign.jpg`
the two models detect different things.

`bus.jpg` and `people.jpg` are the sample images that ship with Ultralytics. The others are from the
[COCO 2017 validation set](https://cocodataset.org/).

## Endpoints

| Method | Path             | Description                                              |
|--------|------------------|----------------------------------------------------------|
| GET    | `/health`        | Health check                                             |
| GET    | `/models`        | List loaded models and which one is active               |
| GET    | `/model`         | Path, task, device and class names (`?model=` optional)  |
| PUT    | `/model`         | Switch the active model: `{"name": "yolo11s"}`           |
| POST   | `/predict`       | Upload an image → JSON list of detections                |
| POST   | `/predict/image` | Upload an image → JPEG with boxes drawn on it            |

Both `/predict` endpoints accept optional `conf` (confidence threshold), `iou` (NMS threshold) and `model`
(model name; defaults to the active model) query params. The JSON response includes a `model` field and the
image response an `X-Model` header, so you can see which model answered.

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

## Adding your own model

Every `.pt` file in the `models/` folder is served, and the folder is mounted into the container, so you can
add models while `docker compose up` is running.

1. Copy your YOLO weights into `models/`.
2. The model is named after its file, so `yolo11n.pt` becomes `yolo11n`.