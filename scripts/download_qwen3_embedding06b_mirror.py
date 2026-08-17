from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


FILES = [
    ".gitattributes",
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def download_file(url: str, dst: Path, *, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    headers = {}
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        if resp.status_code == 416:
            tmp.rename(dst)
            return
        resp.raise_for_status()
        mode = "ab" if existing and resp.status_code == 206 else "wb"
        with tmp.open(mode) as handle:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    tmp.rename(dst)


def download_file_with_retries(url: str, dst: Path, *, retries: int = 50) -> None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            download_file(url, dst)
            return
        except Exception as exc:
            last_err = exc
            part = dst.with_suffix(dst.suffix + ".part")
            size = part.stat().st_size if part.exists() else 0
            wait = min(30, 2 * attempt)
            print(f"[warn] {dst.name} attempt={attempt} failed after {size} bytes: {exc}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output-dir", default="/home/sunyuwei/LightMem/LightMem/models/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    out = Path(args.output_dir)
    for rel in FILES:
        dst = out / rel
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {rel} ({dst.stat().st_size} bytes)", flush=True)
            continue
        url = f"{args.endpoint.rstrip('/')}/{args.repo}/resolve/main/{rel}"
        print(f"[download] {rel}", flush=True)
        download_file_with_retries(url, dst)
        print(f"[done] {rel} ({dst.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
