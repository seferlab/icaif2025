
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(obj: Any, path: str) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
