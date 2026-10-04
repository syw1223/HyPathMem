from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from common import ROOT


PYTHON = sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LongMemEval-S V3.9 migration runner. This is intentionally dataset-aware; "
            "it does not reuse LoCoMo category filtering or train a selector on LongMemEval-S by default."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "nodes", "graph", "semantic-v3", "sanity", "all-sanity", "all-base"],
        default="all-sanity",
    )
    parser.add_argument("--config", default="configs/longmemeval_s.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage == "all-sanity":
        stages = ["prepare", "nodes", "sanity"]
    elif args.stage == "all-base":
        stages = ["prepare", "nodes", "graph", "semantic-v3", "sanity"]
    else:
        stages = [args.stage]
    for stage in stages:
        command = command_for_stage(stage, args)
        run(command, dry_run=args.dry_run)


def command_for_stage(stage: str, args: argparse.Namespace) -> list[str]:
    if stage == "prepare":
        command = [
            PYTHON,
            "scripts/00_prepare_longmemeval.py",
            "--config",
            args.config,
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        return command
    if stage == "nodes":
        return [
            PYTHON,
            "scripts/01_extract_nodes.py",
            "--config",
            args.config,
        ]
    if stage == "graph":
        return [
            PYTHON,
            "scripts/02_build_relations.py",
            "--config",
            args.config,
            "--graph-id",
            "longmemeval_s_base_graph",
            "--light-support-only",
        ]
    if stage == "semantic-v3":
        return [
            PYTHON,
            "scripts/54_build_semantic_hierarchy_v3.py",
            "--config",
            args.config,
            "--graph",
            "outputs/longmemeval_s/graph_v3_9.json",
            "--output",
            "outputs/longmemeval_s/graph_semantic_hierarchy_v3.json",
            "--diagnostics",
            "outputs/eval/longmemeval_semantic_hierarchy_v3_diagnostics.json",
            "--graph-id",
            "longmemeval_s_semantic_hierarchy_v3",
            "--rule-fact-policy",
            "all",
        ]
    if stage == "sanity":
        return [
            PYTHON,
            "-c",
            (
                "import json; "
                "p='data/longmemeval/processed/longmemeval_s_mvp.json'; "
                "d=json.load(open(p,encoding='utf-8')); "
                "print({'instances':len(d),'turns':sum(len(x['turns']) for x in d),"
                "'gold_turns':sum(len(x['qa'][0]['gold_evidence']) for x in d),"
                "'abstention':sum(bool(x['qa'][0]['is_abstention']) for x in d)})"
            ),
        ]
    raise ValueError(f"unknown stage: {stage}")


def run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
