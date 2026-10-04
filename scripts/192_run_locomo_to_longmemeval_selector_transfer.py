"""Train on LoCoMo Cat1-4, freeze, and evaluate LME retrieval on CPU."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    spec = importlib.util.spec_from_file_location("transfer_" + name.split("_")[0], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inference_items(items):
    # Labels remain in the evaluation records, never in the scorer's query objects.
    return [{"question_id": item["question_id"], "question": item["question"],
             "paths": item["paths"]} for item in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/analysis/locomo_to_longmemeval_selector_transfer_20260922")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_selector.txt").exists():
        raise RuntimeError("Output already contains a frozen model; use a new output directory")
    audit = load("191_run_selector_protocol_audit.py")
    cv = load("30_run_loco_cv_selector.py")
    lme = load("120_run_longmemeval_24_feature_card_selector.py")
    loco = load("108_run_v3_9_24_feature_card_quota_cv.py")
    from hytopomem.memory.graph_store import JsonGraphStore
    from hytopomem.retrieval.topology_features import TopologyFeatureIndex, extract_topology_features, with_cached_query_terms, project_feature_rows
    from hytopomem.retrieval.topology_selector import train_lightgbm_ranker
    import lightgbm

    assert loco.RETRIEVAL_ROUTE_CARD5 == lme.RETRIEVAL_ROUTE_CARD5
    features = list(loco.RETRIEVAL_ROUTE_CARD5)
    params = dict(n_estimators=80, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                  n_jobs=8, random_state=13)
    started = time.time()
    source_path = ROOT / "outputs/eval/cv/v3_9_24_feature_card_guided_expand120/card_quota_24_feature_summary.json"
    source = audit.read(source_path)
    manifest = {"protocol": "LoCoMo Cat1-4 supervised training; frozen LongMemEval-S transfer",
                "parameters": params, "features": features, "lightgbm_version": lightgbm.__version__,
                "script_sha256": audit.sha(__file__), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "training_sources": {key: {"path": source[key], "sha256": audit.sha(source[key])}
                                     for key in ("graph", "candidates")},
                "limitations": "Post-hoc transfer test. Upstream artifacts were developed previously; not an untouched benchmark. No QA or judge is run."}
    audit.save(out / "manifest.json", manifest)
    print("Loading LoCoMo training graph and Cat1-4 candidates", flush=True)
    graph = JsonGraphStore().load(source["graph"])
    train = [x for x in audit.read(source["candidates"]) if str(x.get("category")) in {"1", "2", "3", "4"}]
    qids = [x["question_id"] for x in train]
    assert len(qids) == len(set(qids)) == 1540
    assert len({cv.conversation_id_from_question(q) for q in qids}) == 10
    cache = cv.build_feature_cache(graph, train)
    rows, labels, groups, kept = cv.flatten_feature_cache(cache, qids)
    assert sum(groups) == len(rows) == len(labels)
    audit.save(out / "training_manifest.json", {"qids": kept, "questions": len(kept),
               "rows": len(rows), "positive_rows": sum(labels), "negative_rows": len(labels)-sum(labels),
               "no_gold_questions": sum(not x.get("gold_evidence") for x in train),
               "all_negative_groups": sum(not any(cache[q]["labels"]) for q in kept)})
    print(f"Training frozen selector: questions={len(kept)}, rows={len(rows)}, parameters={params}", flush=True)
    model = train_lightgbm_ranker(project_feature_rows(rows, features), labels, groups, **params)
    model.booster_.save_model(str(out / "frozen_selector.txt"))
    frozen_hash = audit.sha(out / "frozen_selector.txt")
    audit.save(out / "frozen_model.json", {"sha256": frozen_hash, "features": features, "parameters": params,
                                          "frozen_at": time.time()})
    del graph, train, cache, rows, labels, groups
    gc.collect()

    test_source_path = ROOT / "outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary_top50/summary.json"
    test_source = audit.read(test_source_path)
    manifest["test_sources"] = {key: {"path": test_source[key], "sha256": audit.sha(test_source[key])}
                                for key in ("graph", "candidates", "cardce_paths")}
    manifest["candidate_topn"] = test_source["candidate_topn"]
    manifest["card_quota"] = 5
    manifest["evaluation_k"] = [20, 50]
    manifest["model_sha256"] = frozen_hash
    audit.save(out / "manifest.json", manifest)
    print("Frozen model saved. Loading LongMemEval graph; no further fitting", flush=True)
    graph = JsonGraphStore().load(test_source["graph"])
    evaluation = lme.truncate_candidates(lme.canonicalize_longmemeval_candidates(audit.read(test_source["candidates"])), test_source["candidate_topn"])
    test_qids = {x["question_id"] for x in evaluation}
    assert len(evaluation) == len(test_qids) == 500
    assert not set(kept) & test_qids
    test = inference_items(evaluation)
    index = TopologyFeatureIndex.from_graph(graph)
    cache = {}
    for i, item in enumerate(test):
        query = with_cached_query_terms(item)
        cache[item["question_id"]] = {"rows": [extract_topology_features(graph, query, p, r, index)
                                                for r, p in enumerate(item["paths"], 1)]}
        if (i+1) % 100 == 0:
            print(f"Label-free inference features: {i+1}/500", flush=True)
    scores = cv.score_items_with_cache(test, model, cache, features)
    audit.save(out / "frozen_scores.json", scores)
    card_index = lme.build_cardce_index(inference_items(audit.read(test_source["cardce_paths"])))
    selected = lme.card_quota_items(test, scores, card_index, topk=50, card_quota=5, method="LoCoMo_frozen_transfer")
    audit.save(out / "label_free_top50_paths.json", selected)
    assert audit.sha(out / "frozen_selector.txt") == frozen_hash

    # Attach labels only after all rankings have been frozen and written.
    evaluation_by_qid = {x["question_id"]: x for x in evaluation}
    selected = [{**evaluation_by_qid[x["question_id"]], "paths": x["paths"], "metadata": x.get("metadata", {})}
                for x in selected]
    original = audit.read(test_source_path.parent / "card_quota_light_top50_paths.json")
    assert {x["question_id"] for x in original} == test_qids
    metrics = []
    for k in (20, 50):
        for name, items in (("Original_inbenchmark_CV", original),
                            ("Same_pool_CE", audit.ce_items(evaluation, k)),
                            ("LoCoMo_frozen_transfer", selected)):
            paths = [{**x, "paths": x["paths"][:k]} for x in items]
            metrics.append(audit.evaluate(graph, paths, k, name, out))
    audit.save(out / "summary.json", {"status": "completed", "seconds": time.time()-started,
                                      "metrics": metrics, "model_sha256": frozen_hash})
    lines = ["# LoCoMo-trained frozen selector on LongMemEval-S", "",
             "500 questions retained. Retrieval metrics below use the subset with annotated evidence; all-question diagnostics are in JSON.",
             "No generation/judge or GPU calls. Original is the historical in-benchmark CV comparison, not a clean transfer result.", "",
             "| Protocol | K | Questions with gold | Hit (%) | Recall (%) | FullCover (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in metrics:
        s = m["with_gold"]
        lines.append(f"| {m['protocol']} | {m['k']} | {s['num_questions']} | {100*s['hit']:.4f} | {100*s['recall']:.4f} | {100*s['full_cover']:.4f} |")
    lines += ["", "This post-hoc experiment removes LongMemEval labels from selector fitting, not prior benchmark-informed development."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
    audit.save(out / "completion.json", {"status": "completed", "questions": 500, "model_unchanged": True})
    print("COMPLETED: frozen transfer retrieval @20/@50", flush=True)


if __name__ == "__main__":
    main()
