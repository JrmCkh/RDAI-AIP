import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_models(raw: str) -> dict[str, str]:
    """Parse MODELS="name=path,path2,..." into {name: path}.

    An entry without "name=" is named after its file stem (e.g. "yolo11s.pt" -> "yolo11s").
    """
    models: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        name, sep, path = entry.partition("=")
        if not sep:
            name, path = Path(entry).stem, entry
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid MODELS entry: {entry!r}")
        if name in models:
            raise ValueError(f"Duplicate model name in MODELS: {name!r}")
        models[name] = path
    if not models:
        raise ValueError("MODELS must list at least one model.")
    return models


@dataclass(frozen=True)
class Settings:
    models: dict[str, str] = field(
        default_factory=lambda: _parse_models(os.getenv(
            "MODELS", "yolo11n=models/yolo11n.pt,yolo11s=models/yolo11s.pt"
        ))
    )
    default_model: str = os.getenv("DEFAULT_MODEL", "")
    device: str = os.getenv("DEVICE", "cpu")
    default_conf: float = float(os.getenv("DEFAULT_CONF", "0.25"))
    default_iou: float = float(os.getenv("DEFAULT_IOU", "0.45"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "10"))

    def __post_init__(self):
        if not self.default_model:
            object.__setattr__(self, "default_model", next(iter(self.models)))
        if self.default_model not in self.models:
            raise ValueError(
                f"DEFAULT_MODEL {self.default_model!r} is not in MODELS ({', '.join(self.models)})."
            )


settings = Settings()
