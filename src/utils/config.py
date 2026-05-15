from __future__ import annotations
from pathlib import Path
import json, yaml

def load_yaml(path: str|Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def save_json(obj: dict, path: str|Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=2, default=str)
