from __future__ import annotations

import argparse
import os
import shlex
import subprocess

from common import ROOT


PYTHON = "/home/sunyuwei/miniconda3/envs/python311/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval V3.10 fine-grained FACT + EVENT/EPISODE/TOPIC semantic hierarchy."
    )
    parser.add_argument(
        "--stage",
        choices=["nodes", "graph", "semantic", "episode", "qwen-summary", "all-no-qwen", "all"],
        default="all-no-qwen",
    )
    parser.add_argument("--config", default="configs/longmemeval_s.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--max-sentences-per-turn", type=int, default=6)
    parser.add_argument("--event-threshold", type=float, default=0.35)
    parser.add_argument("--topic-threshold", type=float, default=0.42)
    parser.add_argument("--episode-threshold", type=float, default=0.50)
    parser.add_argument("--qwen-scope", choices=["all", "path_nodes"], default="all")
    parser.add_argument("--qwen-paths", default="outputs/eval/longmemeval_v3_9_24_feature_card_selector/lgbm_top20_paths.json")
    parser.add_argument("--qwen-path-topn", type=int, default=20)
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--limit-topics", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stages = {
        "all-no-qwen": ["nodes", "graph", "semantic", "episode"],
        "all": ["nodes", "graph", "semantic", "episode", "qwen-summary"],
    }.get(args.stage, [args.stage])
    for stage in stages:
        command = command_for_stage(stage, args)
        run(command, dry_run=args.dry_run)


def command_for_stage(stage: str, args: argparse.Namespace) -> list[str]:
    prefix = "outputs/longmemeval_s/v3_10_fine"
    limit_suffix = f"_first{args.limit}" if args.limit else ""
    nodes = f"{prefix}/nodes_sentence_facts_cap{args.max_sentences_per_turn}{limit_suffix}.json"
    graph = f"{prefix}/graph_sentence_facts_cap{args.max_sentences_per_turn}{limit_suffix}.json"
    semantic = f"{prefix}/graph_sentence_semantic_v3_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.json"
    episode = f"{prefix}/graph_sentence_semantic_episode_v3_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.json"
    qwen = f"{prefix}/graph_sentence_semantic_episode_qwen_v3_10_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.json"
    if stage == "nodes":
        command = [
            PYTHON,
            "scripts/123_build_longmemeval_finegrained_nodes.py",
            "--config",
            args.config,
            "--max-sentences-per-turn",
            str(args.max_sentences_per_turn),
            "--output",
            nodes,
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        return command
    if stage == "graph":
        return [
            PYTHON,
            "scripts/02_build_relations.py",
            "--nodes",
            nodes,
            "--output",
            graph,
            "--graph-id",
            f"longmemeval_s_v3_10_sentence_fact_cap{args.max_sentences_per_turn}{limit_suffix}",
            "--light-support-only",
        ]
    if stage == "semantic":
        command = [
            PYTHON,
            "scripts/54_build_semantic_hierarchy_v3.py",
            "--graph",
            graph,
            "--output",
            semantic,
            "--diagnostics",
            f"outputs/eval/longmemeval_v3_10_sentence_semantic_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.json",
            "--graph-id",
            f"longmemeval_s_v3_10_sentence_semantic_cap{args.max_sentences_per_turn}{limit_suffix}",
            "--rule-fact-policy",
            "all",
            "--event-similarity-threshold",
            str(args.event_threshold),
            "--topic-similarity-threshold",
            str(args.topic_threshold),
            "--event-max-facts",
            "6",
            "--topic-max-events",
            "8",
            "--embedding-batch-size",
            str(args.embedding_batch_size),
        ]
        if args.device:
            command.extend(["--device", args.device])
        return command
    if stage == "episode":
        command = [
            PYTHON,
            "scripts/64_build_episode_hierarchy_v3_3.py",
            "--graph",
            semantic,
            "--output",
            episode,
            "--diagnostics",
            f"outputs/eval/longmemeval_v3_10_sentence_episode_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.json",
            "--graph-id",
            f"longmemeval_s_v3_10_sentence_episode_cap{args.max_sentences_per_turn}{limit_suffix}",
            "--episode-similarity-threshold",
            str(args.episode_threshold),
            "--episode-max-events",
            "6",
            "--min-episode-events",
            "2",
            "--embedding-batch-size",
            str(args.embedding_batch_size),
        ]
        if args.device:
            command.extend(["--device", args.device])
        return command
    if stage == "qwen-summary":
        command = [
            PYTHON,
            "scripts/124_annotate_longmemeval_event_topic_qwen.py",
            "--graph",
            episode,
            "--output",
            qwen,
            "--cache",
            f"outputs/llm_annotations/longmemeval_v3_10_qwen_summary_cap{args.max_sentences_per_turn}_t{int(args.event_threshold * 100):03d}{limit_suffix}.jsonl",
            "--scope",
            args.qwen_scope,
            "--path-topn",
            str(args.qwen_path_topn),
            "--resume",
        ]
        if args.qwen_scope == "path_nodes":
            command.extend(["--paths", args.qwen_paths])
        if args.limit_events:
            command.extend(["--limit-events", str(args.limit_events)])
        if args.limit_episodes:
            command.extend(["--limit-episodes", str(args.limit_episodes)])
        if args.limit_topics:
            command.extend(["--limit-topics", str(args.limit_topics)])
        return command
    raise ValueError(f"unknown stage: {stage}")


def run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
