import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "yolo11n.pt")
    device: str = os.getenv("DEVICE", "cpu")
    default_conf: float = float(os.getenv("DEFAULT_CONF", "0.25"))
    default_iou: float = float(os.getenv("DEFAULT_IOU", "0.45"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "10"))


settings = Settings()
