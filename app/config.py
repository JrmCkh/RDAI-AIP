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


def _discover_models(models_dir: str) -> dict[str, str]:
    """Find every *.pt file in models_dir, named after its file stem (e.g. "best.pt" -> "best")."""
    folder = Path(models_dir)
    models = {p.stem: str(p) for p in sorted(folder.glob("*.pt"))}
    if not models:
        raise ValueError(f"No .pt model files found in {folder.resolve()} (set MODELS_DIR or MODELS).")
    return models


@dataclass(frozen=True)
class Settings:
    # Folder scanned for *.pt files when MODELS isn't set. Drop a .pt file in here to add a model.
    models_dir: str = os.getenv("MODELS_DIR", "models")
    # Optional explicit "name=path,..." list; when set, it replaces folder discovery.
    models_env: str = os.getenv("MODELS", "")
    models: dict[str, str] = field(init=False)
    # Model used when a request names none; empty = first model that loads successfully.
    default_model: str = os.getenv("DEFAULT_MODEL", "")
    device: str = os.getenv("DEVICE", "cpu")
    default_conf: float = float(os.getenv("DEFAULT_CONF", "0.25"))
    default_iou: float = float(os.getenv("DEFAULT_IOU", "0.45"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "10"))

    @property
    def models_discovered(self) -> bool:
        return not self.models_env

    def __post_init__(self):
        models = _parse_models(self.models_env) if self.models_env else _discover_models(self.models_dir)
        object.__setattr__(self, "models", models)
        # Empty default_model means "first model that loads successfully" (resolved at startup).
        if self.default_model and self.default_model not in self.models:
            raise ValueError(
                f"DEFAULT_MODEL {self.default_model!r} is not in MODELS ({', '.join(self.models)})."
            )


settings = Settings()
