"""Restore hash-verified gzip-split artifacts into repository-relative paths."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes release directory: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(root: Path, artifact: dict) -> Path:
    target = safe_path(root, artifact["destination"])
    if target.exists():
        if sha256(target) != artifact["sha256"]:
            raise ValueError(f"Refusing to overwrite different existing artifact: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.TemporaryFile() as compressed:
            for part in artifact["parts"]:
                path = safe_path(root, part["path"])
                if path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
                    raise ValueError(f"Corrupt artifact part: {path}")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        compressed.write(chunk)
            compressed.seek(0)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with gzip.GzipFile(fileobj=compressed, mode="rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(chunk)
        if temporary.stat().st_size != artifact["bytes"] or sha256(temporary) != artifact["sha256"]:
            raise ValueError(f"Restored artifact digest mismatch: {target}")
        temporary.replace(target)
        return target
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.root / "results/external/manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        print(f"verified and restored: {restore(args.root, artifact).relative_to(args.root.resolve())}")


if __name__ == "__main__":
    main()
