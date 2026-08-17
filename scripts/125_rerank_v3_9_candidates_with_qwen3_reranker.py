from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, summarize
from hytopomem.memory.graph_store import JsonGraphStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank V3.9 candidate paths with a Qwen3/BGE-style CrossEncoder reranker."
    )
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_guided_expand120.json")
    parser.add_argument(
        "--baseline",
        default="outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json",
        help="Existing CE+LightGBM/card-quota top20 path file for comparison.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--candidate-topn", type=int, default=0, help="Score first N candidates per query; 0=all.")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/eval/v3_9_qwen3_reranker8b_top20")
    args = parser.parse_args()

    started = time.time()
    graph = JsonGraphStore().load(resolve_path(args.graph))
    candidates = read_json(resolve_path(args.candidates))
    baseline = read_json(resolve_path(args.baseline)) if args.baseline else []
    if args.max_questions:
        candidates = candidates[: args.max_questions]
        baseline = baseline[: args.max_questions]
    if args.candidate_topn:
        candidates = truncate_candidates(candidates, args.candidate_topn)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"loaded candidates questions={len(candidates)} model={args.model} "
        f"candidate_topn={args.candidate_topn or 'all'}",
        flush=True,
    )

    reranker = Qwen3CausalReranker(args.model, args.device, max_length=args.max_length)
    checkpoint_path = output_dir / "qwen3_reranker_paths.jsonl"
    reranked, score_stats = rerank_items(graph, candidates, reranker, args.batch_size, checkpoint_path)
    write_json(reranked, output_dir / "qwen3_reranker_paths.json")

    summary = {
        "method": "Qwen3 reranker-only over V3.9 expand120 candidates",
        "model": args.model,
        "device": args.device,
        "graph": str(resolve_path(args.graph)),
        "candidates": str(resolve_path(args.candidates)),
        "baseline": str(resolve_path(args.baseline)) if args.baseline else "",
        "candidate_topn": args.candidate_topn or None,
        "num_questions": len(candidates),
        "score_stats": score_stats,
        "aggregate": {},
        "paired_hit": {},
        "elapsed_sec": time.time() - started,
    }
    for k in args.topk:
        rerank_eval = evaluate_items(graph, reranked, k, f"qwen3_reranker_top{k}")
        write_json(rerank_eval, output_dir / f"qwen3_reranker_top{k}_eval.json")
        summary["aggregate"][f"qwen3_reranker_top{k}"] = rerank_eval["summary"]
        if baseline:
            base_eval = evaluate_items(graph, baseline, k, f"baseline_top{k}")
            write_json(base_eval, output_dir / f"baseline_top{k}_eval.json")
            summary["aggregate"][f"baseline_top{k}"] = base_eval["summary"]
            summary["paired_hit"][f"qwen3_vs_baseline_top{k}"] = paired_hit(
                rerank_eval["per_question"], base_eval["per_question"]
            )

    write_json(summary, output_dir / "summary.json")
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary), flush=True)


class Qwen3CausalReranker:
    """Qwen3-Reranker official yes/no next-token scorer.

    The model ships as a SentenceTransformers CrossEncoder wrapper, but loading it
    through AutoModelForSequenceClassification can initialize a fresh classifier
    head in some local environments. This class follows the Transformers example
    in the model README and scores P("yes" | query, document).
    """

    def __init__(self, model_name_or_path: str, device: str, max_length: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.instruction = "Given a memory question, retrieve relevant memory facts or passages that answer the question"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int, show_progress_bar: bool = True) -> list[float]:
        from tqdm import tqdm

        scores: list[float] = []
        iterator = range(0, len(pairs), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Qwen3 rerank batches")
        for start in iterator:
            batch = pairs[start : start + batch_size]
            texts = [self.format_instruction(query, doc) for query, doc in batch]
            inputs = self.process_inputs(texts)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                true_vector = logits[:, self.token_true_id]
                false_vector = logits[:, self.token_false_id]
                two_class = self.torch.stack([false_vector, true_vector], dim=1)
                probs = self.torch.nn.functional.log_softmax(two_class, dim=1)[:, 1].exp()
            scores.extend(float(score) for score in probs.detach().cpu().tolist())
        return scores

    def format_instruction(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, texts: list[str]):
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for idx, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = self.prefix_tokens + input_ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        return {key: value.to(self.model.device) for key, value in inputs.items()}


def truncate_candidates(items: list[dict], topn: int) -> list[dict]:
    out = []
    for item in items:
        copied = dict(item)
        copied["paths"] = list(item.get("paths", []))[:topn]
        metadata = dict(copied.get("metadata", {}))
        metadata["qwen_reranker_candidate_topn"] = topn
        copied["metadata"] = metadata
        out.append(copied)
    return out


def rerank_items(graph, items: list[dict], reranker, batch_size: int, checkpoint_path: Path) -> tuple[list[dict], dict]:
    from tqdm import tqdm

    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("question_id") or "")
                if qid:
                    done[qid] = row

    total_pairs = sum(len(item.get("paths", [])) for item in items)
    print(
        f"scoring questions={len(items)} pairs={total_pairs} batch_size={batch_size} "
        f"checkpoint_done={len(done)} checkpoint={checkpoint_path}",
        flush=True,
    )

    all_scores: list[float] = []
    output_by_qid: dict[str, dict] = {}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as ckpt:
        for item in tqdm(items, desc="Qwen3 rerank questions"):
            qid = str(item.get("question_id") or "")
            if qid in done:
                output_by_qid[qid] = done[qid]
                for path in done[qid].get("paths", []):
                    if "score" in path:
                        all_scores.append(float(path["score"]))
                continue

            question = str(item.get("question") or "")
            pairs = [(question, path_text(graph, path)) for path in item.get("paths", [])]
            scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False) if pairs else []
            all_scores.extend(float(score) for score in scores)
            ranked = sorted(enumerate(scores), key=lambda row: row[1], reverse=True)

            copied = dict(item)
            selected = []
            for rank, (path_idx, score) in enumerate(ranked, start=1):
                path = dict(item["paths"][path_idx])
                path["score"] = float(score)
                scores_map = dict(path.get("scores", {}))
                scores_map["qwen3_reranker"] = float(score)
                path["scores"] = scores_map
                metadata = dict(path.get("metadata", {}))
                metadata["qwen3_reranker_rank"] = str(rank)
                metadata["qwen3_reranker_model"] = "Qwen3-Reranker-8B"
                path["metadata"] = metadata
                selected.append(path)
            copied["paths"] = selected
            metadata = dict(copied.get("metadata", {}))
            metadata["method"] = "qwen3_reranker_only"
            copied["metadata"] = metadata
            output_by_qid[qid] = copied
            ckpt.write(json.dumps(copied, ensure_ascii=False) + "\n")
            ckpt.flush()

    output = []
    for item in items:
        qid = str(item.get("question_id") or "")
        output.append(output_by_qid[qid])

    return output, {
        "num_pairs": len(all_scores),
        "min": min(all_scores) if all_scores else 0.0,
        "max": max(all_scores) if all_scores else 0.0,
        "mean": mean(all_scores) if all_scores else 0.0,
    }


def path_text(graph, path: dict) -> str:
    evidence_id = path.get("metadata", {}).get("evidence_node_id")
    node_ids = [evidence_id] if evidence_id else []
    node_ids.extend(path.get("node_ids", []))
    seen = set()
    parts = []
    for node_id in node_ids:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = graph.nodes.get(str(node_id))
        if node is None:
            continue
        node_type = getattr(node.type, "value", str(node.type))
        parts.append(f"[{node_type}] {node.text}")
    if parts:
        return "\n".join(parts)
    return str(path.get("metadata", {}).get("v3_9_card_summary") or path.get("metadata", {}).get("candidate_source") or "")


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    return {
        "method": method,
        "k": k,
        "summary": summarize(results),
        "per_question": [result.__dict__ for result in results],
    }


def paired_hit(left_rows: list[dict], right_rows: list[dict]) -> dict:
    right = {row["question_id"]: row for row in right_rows}
    both_hit = left_only = right_only = both_miss = compared = 0
    for row in left_rows:
        other = right.get(row["question_id"])
        if other is None:
            continue
        compared += 1
        left_hit = bool(row["hit"])
        right_hit = bool(other["hit"])
        if left_hit and right_hit:
            both_hit += 1
        elif left_hit:
            left_only += 1
        elif right_hit:
            right_only += 1
        else:
            both_miss += 1
    return {
        "compared": compared,
        "both_hit": both_hit,
        "qwen_only": left_only,
        "baseline_only": right_only,
        "both_miss": both_miss,
        "net_qwen_minus_baseline": left_only - right_only,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Qwen3 Reranker Top-K Comparison",
        "",
        f"Model: `{summary['model']}`",
        f"Candidates: `{summary['candidates']}`",
        f"Baseline: `{summary['baseline']}`",
        "",
        "## Aggregate",
        "",
        "| Method | N | Hit | Recall | FullCover | AvgTokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["aggregate"].items():
        lines.append(
            f"| {method} | {metrics['num_questions']} | {metrics['hit']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['full_cover']:.4f} | {metrics['avg_tokens']:.1f} |"
        )
    if summary["paired_hit"]:
        lines.extend(["", "## Paired Hit", "", "| Compare | Compared | QwenOnly | BaselineOnly | Net |", "|---|---:|---:|---:|---:|"])
        for name, row in summary["paired_hit"].items():
            lines.append(
                f"| {name} | {row['compared']} | {row['qwen_only']} | "
                f"{row['baseline_only']} | {row['net_qwen_minus_baseline']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
