from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import ROOT


PYTHON = sys.executable
QWEN_MODEL = "qwen3-30b-a3b-instruct-2507"
QWEN_URL = "http://127.0.0.1:8006/v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Packaged LoCoMo V3.9 mainline runner. Defaults are the frozen mainline paths."
    )
    parser.add_argument(
        "--stage",
        choices=["cards", "expand", "selector", "qa-hybrid", "qa-hybrid-gpt41mini", "all"],
        default="qa-hybrid",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stages = ["cards", "expand", "selector", "qa-hybrid"] if args.stage == "all" else [args.stage]
    for stage in stages:
        command = command_for_stage(stage, args)
        run(command, dry_run=args.dry_run)


def command_for_stage(stage: str, args: argparse.Namespace) -> list[str]:
    if stage == "cards":
        return [
            PYTHON,
            "scripts/94_build_v3_9_query_conditioned_relation_cards.py",
            "--graph",
            "outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
            "--base-paths",
            "outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
            "--base-topn",
            "150",
            "--context-topn",
            "50",
            "--model",
            QWEN_MODEL,
            "--base-url",
            QWEN_URL,
            "--api-key-env",
            "VLLM_API_KEY",
            "--base-url-env",
            "VLLM_BASE_URL",
            "--workers",
            str(args.workers),
            "--resume",
            "--cache",
            "outputs/v3_9_query_cards/qwen3_cards_v3.jsonl",
            "--output",
            "outputs/v3_9_query_cards/qwen3_card_annotated_base150_paths_v3_clean.json",
            "--summary-json",
            "outputs/eval/V3_9_QUERY_CARD_BASE150_SUMMARY_V3_CLEAN.json",
            "--summary-md",
            "outputs/eval/V3_9_QUERY_CARD_BASE150_SUMMARY_V3_CLEAN.md",
        ]
    if stage == "expand":
        return [
            PYTHON,
            "scripts/109_build_v3_9_card_guided_local_expansion.py",
            "--input",
            "outputs/v3_9_query_cards/qwen3_card_annotated_base150_paths_v3_clean.json",
            "--base-topn",
            "100",
            "--extra",
            "20",
            "--output-prefix",
            "outputs/v3_9_query_cards/qwen3_card_guided_expand",
        ]
    if stage == "selector":
        return [
            PYTHON,
            "scripts/108_run_v3_9_24_feature_card_quota_cv.py",
            "--graph",
            "outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
            "--candidates",
            "outputs/v3_9_query_cards/qwen3_card_guided_expand120.json",
            "--output-dir",
            "outputs/eval/cv/v3_9_24_feature_card_guided_expand120",
            "--topk",
            "5",
            "20",
        ]
    if stage in {"qa-hybrid", "qa-hybrid-gpt41mini"}:
        model = "gpt-4.1-mini" if stage == "qa-hybrid-gpt41mini" else "gpt-4o-mini"
        judge_model = "gpt-4.1-mini" if stage == "qa-hybrid-gpt41mini" else "gpt-4o-mini"
        suffix = "gpt41mini_judge_gpt41mini" if stage == "qa-hybrid-gpt41mini" else "gpt4omini"
        return [
            PYTHON,
            "scripts/06_run_qa_eval.py",
            "--graph",
            "outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
            "--paths",
            "outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json",
            "--output",
            f"outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_chatanywhere_{suffix}.json",
            "--log",
            f"outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_chatanywhere_{suffix}.log",
            "--k",
            "20",
            "--categories",
            "1,2,3,4",
            "--model",
            model,
            "--judge-model",
            judge_model,
            "--context-mode",
            "hybrid",
            "--answer-protocol",
            "default",
            "--verify-answer",
            "none",
            "--save-every",
            "1",
            "--resume",
        ]
    raise ValueError(f"unknown stage: {stage}")


def run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("VLLM_API_KEY", "EMPTY")
    env.setdefault("VLLM_BASE_URL", QWEN_URL)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
