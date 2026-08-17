from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore


SETTINGS = {
    "base100": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_type_specific.json",
        "candidates": "outputs/nary_v3_6c_selector/base100_paths.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_base100_top20",
    },
    "qwen_type_completion20": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_type_specific.json",
        "candidates": "outputs/nary_v3_6c_selector/qwen_type_specific_base100_completion20_paths.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_qwen_type_completion20_top20",
    },
    "gpt4o_clean_completion20": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_gpt4o_clean.json",
        "candidates": "outputs/nary_v3_6c_selector/gpt4o_clean_base100_completion20_paths.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_gpt4o_clean_completion20_top20",
    },
    "qwen_all_completion50": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
        "candidates": "outputs/nary_v3_6c_selector/qwen_all_base100_completion50_paths.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_qwen_all_completion50_top20",
    },
}

METHODS = [
    "candidate_ce_input_order",
    "base_completion_no_nary_features",
    "base_completion_nary_point_features",
    "base_completion_nary_point_set_features",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/eval/NARY_V3_6C_SELECTOR_COMPLETION_SUMMARY.json")
    parser.add_argument("--output-md", default="outputs/eval/NARY_V3_6C_SELECTOR_COMPLETION_SUMMARY.md")
    args = parser.parse_args()

    payload = {"settings": {}}
    for setting_name, setting in SETTINGS.items():
        graph = JsonGraphStore().load(resolve_path(setting["graph"]))
        candidates = read_json(resolve_path(setting["candidates"]))
        cv_dir = resolve_path(setting["cv_dir"])
        setting_payload = {
            "graph": str(resolve_path(setting["graph"])),
            "candidates": str(resolve_path(setting["candidates"])),
            "cv_dir": str(cv_dir),
            "methods": {},
        }
        for method in METHODS:
            if method == "candidate_ce_input_order":
                items = candidates
            else:
                items = load_fold_paths(cv_dir, method)
            setting_payload["methods"][method] = {
                "top5": summarize_results(graph, items, 5),
                "top20": summarize_results(graph, items, 20),
            }
        payload["settings"][setting_name] = setting_payload

    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def load_fold_paths(cv_dir: Path, method: str) -> list[dict]:
    items = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        items.extend(read_json(path))
    if not items:
        raise FileNotFoundError(f"no fold paths found for {method} in {cv_dir}")
    return items


def summarize_results(graph, items: list[dict], k: int) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    summary = summarize(results)
    return {
        "num_questions": summary["num_questions"],
        "hit": summary["hit"],
        "recall": summary["recall"],
        "full_cover": summary["full_cover"],
        "avg_tokens": summary["avg_tokens"],
        "avg_path_len": summary["avg_path_len"],
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.6C N-ary Completion Selector Summary",
        "",
        "## Top-5",
        "",
        "| Setting | Method | Hit@5 | Recall@5 | FullCover@5 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for setting_name, setting in payload["settings"].items():
        for method, metrics_by_k in setting["methods"].items():
            metrics = metrics_by_k["top5"]
            lines.append(
                f"| {setting_name} | {method} | {metrics['hit']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Top-20",
            "",
            "| Setting | Method | Hit@20 | Recall@20 | FullCover@20 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for setting_name, setting in payload["settings"].items():
        for method, metrics_by_k in setting["methods"].items():
            metrics = metrics_by_k["top20"]
            lines.append(
                f"| {setting_name} | {method} | {metrics['hit']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
