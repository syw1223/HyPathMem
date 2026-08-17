#!/usr/bin/env python3
"""Build frozen Top20 D2 raw-grounded packs for LoCoMo."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hytopomem.reconstruction import AnswerPackCompiler, EvidenceUnitBuilder
from hytopomem.reconstruction.locomo_contract import LoCoMoQueryRequirementCompiler


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json"))
    parser.add_argument("--graph", type=Path, default=Path("outputs/graphs/locomo_graph_v3_6b_qwen_all.json"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "outputs/qa/full_v3_9_expand120_light_quota_top20_hybrid_v11_"
            "chatanywhere_gpt41mini_judge_gpt4omini.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction/hypathmem_locomo_d2_v1/top20_structured_raw_packs.json"),
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("paths", "graph", "baseline", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")

    path_rows = read_json(args.paths)
    graph = read_json(args.graph)
    baseline = read_json(args.baseline)
    baseline_rows = baseline.get("per_question") or []
    baseline_by_id = {str(row["question_id"]): row for row in baseline_rows}
    path_by_id = {str(row["question_id"]): row for row in path_rows}
    missing = sorted(set(baseline_by_id) - set(path_by_id))
    if missing:
        raise KeyError(f"Missing frozen paths for {len(missing)} baseline questions: {missing[:5]}")

    selected = baseline_rows[: args.limit or None]
    query_compiler = LoCoMoQueryRequirementCompiler()
    unit_builder = EvidenceUnitBuilder(graph.get("nodes") or {})
    pack_compiler = AnswerPackCompiler(token_budget=args.token_budget)
    rows = []
    for index, base in enumerate(selected, start=1):
        question_id = str(base["question_id"])
        source = path_by_id[question_id]
        contract = query_compiler.compile(str(base["question"]), question_date=None)
        units = unit_builder.build(source.get("paths") or [], contract, k=args.k)
        pack = pack_compiler.compile(str(base["question"]), contract, units)
        rows.append(
            {
                "question_id": question_id,
                "conversation_id": base.get("conversation_id"),
                "question": base["question"],
                "gold_answer": base.get("gold_answer"),
                "category": base.get("category"),
                "question_type": base.get("question_type"),
                "pack": pack.model_dump(mode="json"),
                "answer_context": pack_compiler.render_json(pack),
            }
        )
        if index % 100 == 0 or index == len(selected):
            print(f"built LoCoMo D2 packs {index}/{len(selected)}", flush=True)

    payload = {
        "metadata": {
            "version": "hypathmem_locomo_d2_v1",
            "dataset": "LoCoMo",
            "stage": "frozen_top20_structured_claim_plus_exact_raw_quote",
            "paths": str(args.paths),
            "paths_sha256": sha256(args.paths),
            "graph": str(args.graph),
            "graph_sha256": sha256(args.graph),
            "baseline": str(args.baseline),
            "baseline_sha256": sha256(args.baseline),
            "k": args.k,
            "token_budget": args.token_budget,
            "uses_gold_answer": False,
            "uses_gold_evidence": False,
            "mutates_source_graph": False,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    packs = [row["pack"] for row in rows]
    selected = sum(pack["diagnostics"]["selected_units"] for pack in packs)
    grounded = sum(pack["diagnostics"]["raw_grounded_units"] for pack in packs)
    by_category = {}
    for category in sorted({int(row["category"]) for row in rows}):
        group = [row for row in rows if int(row["category"]) == category]
        by_category[str(category)] = {"n": len(group)}
    return {
        "num_questions": len(rows),
        "supported": sum(pack["answerability"] == "SUPPORTED" for pack in packs),
        "partially_supported": sum(pack["answerability"] == "PARTIALLY_SUPPORTED" for pack in packs),
        "unsupported": sum(pack["answerability"] == "UNSUPPORTED" for pack in packs),
        "selected_units": selected,
        "raw_grounded_units": grounded,
        "raw_grounded_unit_ratio": grounded / selected if selected else 0.0,
        "by_category": by_category,
    }


if __name__ == "__main__":
    main()
