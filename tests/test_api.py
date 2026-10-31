import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics.utils import ASSETS

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _image_bytes(path=ASSETS / "bus.jpg") -> bytes:
    return path.read_bytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": ["yolo11n", "yolo11s"]}


def test_model_info(client):
    r = client.get("/model")
    assert r.status_code == 200
    assert r.json()["name"] == "yolo11n" and r.json()["active"] is True
    assert r.json()["task"] == "detect"
    assert r.json()["num_classes"] == 80


def test_list_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()["active"] == "yolo11n"
    assert [m["name"] for m in r.json()["models"]] == ["yolo11n", "yolo11s"]


def test_predict_with_model_param(client):
    r = client.post(
        "/predict", params={"model": "yolo11s"}, files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")}
    )
    assert r.status_code == 200
    assert r.json()["model"] == "yolo11s"
    assert {"person", "bus"} <= {d["class_name"] for d in r.json()["detections"]}


def test_switch_active_model(client):
    try:
        r = client.put("/model", json={"name": "yolo11s"})
        assert r.status_code == 200 and r.json() == {"active": "yolo11s"}
        r = client.post("/predict", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")})
        assert r.json()["model"] == "yolo11s"
    finally:
        client.put("/model", json={"name": "yolo11n"})


def test_unknown_model_rejected(client):
    assert client.put("/model", json={"name": "nope"}).status_code == 404
    r = client.post("/predict", params={"model": "nope"}, files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")})
    assert r.status_code == 404


def test_predict_detects_people_and_bus(client):
    r = client.post("/predict", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    names = {d["class_name"] for d in body["detections"]}
    assert {"person", "bus"} <= names
    assert body["image_width"] == 810 and body["image_height"] == 1080


def test_predict_conf_threshold_filters(client):
    r = client.post(
        "/predict", params={"conf": 0.99}, files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")}
    )
    assert r.status_code == 200
    assert all(d["confidence"] >= 0.99 for d in r.json()["detections"])


def test_predict_image_returns_jpeg(client):
    r = client.post("/predict/image", files={"file": ("bus.jpg", _image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-model"] == "yolo11n"
    Image.open(io.BytesIO(r.content)).verify()


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
