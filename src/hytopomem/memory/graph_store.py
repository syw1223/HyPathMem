from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from hytopomem.memory.schema import MemoryGraph


PathLike = Union[str, Path]


class JsonGraphStore:
    def save(self, graph: MemoryGraph, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(graph.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)

    def load(self, path: PathLike) -> MemoryGraph:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return MemoryGraph.model_validate(payload)

