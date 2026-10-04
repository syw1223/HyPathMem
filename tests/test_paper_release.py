import argparse
import gzip
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location("release_test_" + name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stage", ["qa-hybrid", "qa-hybrid-gpt41mini"])
def test_paper_generation_and_judge(stage):
    runner = load("run_locomo_v3_9_mainline.py")
    command = runner.command_for_stage(stage, argparse.Namespace(generator_model="gpt-4.1-mini", judge_model="gpt-4o-mini"))
    assert command[command.index("--model") + 1] == "gpt-4.1-mini"
    assert command[command.index("--judge-model") + 1] == "gpt-4o-mini"


def test_qwen_generator_override():
    runner = load("run_locomo_v3_9_mainline.py")
    command = runner.command_for_stage("qa-hybrid", argparse.Namespace(generator_model="qwen3-30b-a3b-instruct-2507", judge_model="gpt-4o-mini"))
    assert command[command.index("--model") + 1] == "qwen3-30b-a3b-instruct-2507"


def test_cat5_filtered_before_training():
    selector = load("108_run_v3_9_24_feature_card_quota_cv.py")
    items = [{"category": c, "question_id": str(i)} for i, c in enumerate([1, "2", 3, 4, 5, "5", None])]
    assert selector.filter_training_categories(items) == items[:4]


def test_artifact_restore_and_corruption(tmp_path):
    restorer = load("restore_release_artifacts.py")
    raw = b'{"test": "artifact"}\n'
    zipped = gzip.compress(raw, mtime=0)
    part = tmp_path / "part.bin"
    part.write_bytes(zipped)
    artifact = {"destination": "outputs/test.json", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "parts": [{"path": part.name, "sha256": hashlib.sha256(zipped).hexdigest(), "bytes": len(zipped)}]}
    target = restorer.restore(tmp_path, artifact)
    assert target.read_bytes() == raw
    assert restorer.restore(tmp_path, artifact) == target
    target.write_bytes(b"different")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        restorer.restore(tmp_path, artifact)
    target.unlink()
    part.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        restorer.restore(tmp_path, artifact)


def test_artifact_path_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        load("restore_release_artifacts.py").safe_path(tmp_path, "../outside.json")
