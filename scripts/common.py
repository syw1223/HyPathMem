from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal environments
    yaml = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_config(path: str | None) -> Dict[str, Any]:
    config_path = Path(path) if path else ROOT / "configs" / "locomo_mvp.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_minimal_yaml(text)


def parse_minimal_yaml(text: str) -> Dict[str, Any]:
    """Tiny parser for the repository's simple nested key/value configs."""
    root: Dict[str, Any] = {}
    current_section: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            root[key] = {}
            current_section = root[key]
            continue
        if current_section is None:
            raise ValueError(f"invalid config line before section: {raw_line}")
        key, value = line.strip().split(":", 1)
        current_section[key] = parse_scalar(value.strip())
    return root


def parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return [parse_scalar(item.strip()) for item in value[1:-1].split(",") if item.strip()]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
