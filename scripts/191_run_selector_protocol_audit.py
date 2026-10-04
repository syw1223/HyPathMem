"""CPU-only, frozen-candidate selector protocol correction; never calls an API."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hytopomem.eval.retrieval_metrics import evaluate_item, summarize, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_features import project_feature_rows
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load(name):
    spec = importlib.util.spec_from_file_location("protocol_" + name.split("_")[0], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_keys(row):
    # Preserve repeated source IDs as separate session instances before hashing.
    sessions = collections.defaultdict(list)
    for turn in row["turns"]:
        match = re.match(r"s(\d+):", str(turn["turn_id"]))
        index = match.group(1) if match else str(turn.get("session_id", ""))
        sessions[index].append(turn)
    keys, records = set(), []
    for turns in sessions.values():
        if not any(str(t.get("text", "")).strip() for t in turns):
            continue
        payload = [[str(t.get("speaker", "")), str(t.get("text", "")).replace("\r\n", "\n").strip()] for t in turns]
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        sid = str(turns[0].get("session_id", ""))
        keys.add("content:" + digest)
        if sid:
            keys.add("id:" + sid)
        records.append((sid, digest))
    return keys, records


def components(keys_by_qid):
    parent = {qid: qid for qid in keys_by_qid}

    def find(qid):
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    owners = {}
    for qid, keys in keys_by_qid.items():
        for key in sorted(keys):
            if key in owners:
                parent[find(qid)] = find(owners[key])
            else:
                owners[key] = qid
    groups = collections.defaultdict(list)
    for qid in sorted(parent):
        groups[find(qid)].append(qid)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def history_audit(out):
    path = ROOT / "data/longmemeval/processed/longmemeval_s_mvp.json"
    print("Loading histories for overlap audit", flush=True)
    histories = read(path)
    keys_by_qid, types = {}, {}
    id_hashes, hash_ids, owners = (collections.defaultdict(set) for _ in range(3))
    for row in histories:
        qid = row["conversation_id"] + ":q0001"
        keys, records = session_keys(row)
        keys_by_qid[qid] = keys
        types[qid] = row.get("question_type", "unknown")
        for sid, digest in records:
            id_hashes[sid].add(digest)
            hash_ids[digest].add(sid)
        for key in keys:
            owners[key].add(qid)
    groups = components(keys_by_qid)
    shared_pairs = set()
    for qids in owners.values():
        values = sorted(qids)
        for i, left in enumerate(values):
            shared_pairs.update((left, right) for right in values[i + 1:])
    # Size-only balancing is label-blind. Never split connected components.
    buckets = [[] for _ in range(min(5, len(groups)))]
    for group in groups:
        buckets[min(range(len(buckets)), key=lambda i: (len(buckets[i]), i))].extend(group)
    usable = len(buckets) >= 2 and max(map(len, buckets)) <= 0.8 * len(histories)
    cv = load("120_run_longmemeval_24_feature_card_selector.py")
    original_folds = cv.make_folds(list(keys_by_qid), 5, 13)

    def fold_audit(folds):
        result = []
        for i, test in enumerate(folds):
            train = sorted(set(keys_by_qid) - set(test))
            train_keys = set().union(*(keys_by_qid[q] for q in train))
            test_keys = set().union(*(keys_by_qid[q] for q in test))
            intersection = train_keys & test_keys
            result.append({"fold": i, "train_qids": train, "test_qids": sorted(test),
                           "shared_source_ids": sum(k.startswith("id:") for k in intersection),
                           "shared_content_hashes": sum(k.startswith("content:") for k in intersection),
                           "test_questions_with_train_overlap": sum(bool(keys_by_qid[q] & train_keys) for q in test)})
        return result

    result = {"dataset": str(path), "sha256": sha(path), "questions": len(histories),
              "identity_policy": "union of exact source session ID and complete ordered role/content hash; empty sessions excluded",
              "component_count": len(groups), "component_sizes": [len(g) for g in groups],
              "shared_source_ids": sum(k.startswith("id:") and len(v) > 1 for k, v in owners.items()),
              "shared_content_hashes": sum(k.startswith("content:") and len(v) > 1 for k, v in owners.items()),
              "overlapping_question_pairs": len(shared_pairs),
              "pair_fraction": len(shared_pairs) / math.comb(len(histories), 2),
              "same_id_different_content": {k: len(v) for k, v in id_hashes.items() if len(v) > 1},
              "same_content_multiple_ids_count": sum(len(v) > 1 for v in hash_ids.values()),
              "groups": [{"group_id": f"history_group_{i:04d}", "qids": g,
                          "categories": dict(collections.Counter(types[q] for q in g))} for i, g in enumerate(groups)],
              "grouped_cv_feasible": usable,
              "feasibility_rule": "at least 2 folds and no test fold larger than 80% of all questions; otherwise do not train",
              "original_random_folds": fold_audit(original_folds), "grouped_folds": fold_audit(buckets)}
    assert all(f["shared_source_ids"] == f["shared_content_hashes"] == 0 for f in result["grouped_folds"])
    save(out / "history_overlap_audit.json", result)
    print(f"History components={len(groups)} largest={len(groups[0])} feasible={usable}", flush=True)
    return result


def ce_items(items, k):
    output = []
    for item in items:
        for path in item["paths"]:
            value = path.get("scores", {}).get("cross_encoder")
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Missing/nonfinite cross_encoder score: {item['question_id']}")
        ranked = sorted(item["paths"], key=lambda p: float(p["scores"]["cross_encoder"]), reverse=True)
        output.append({**item, "paths": ranked[:k]})
    return output


def evaluate(graph, items, k, protocol, out):
    rows = [evaluate_item(graph, item, k) for item in items]
    valid = [row for row in rows if row.gold_evidence_ids]
    result = {"protocol": protocol, "k": k, "all_questions": summarize(rows),
              "with_gold": summarize(valid), "no_gold_questions": len(rows) - len(valid),
              "per_question": [dataclasses.asdict(row) for row in rows]}
    save(out / f"{protocol}_top{k}_metrics.json", result)
    # Preserve rank and complete selected paths for subsequent QA, without calling it.
    save(out / f"{protocol}_top{k}_paths.json", items)
    print(protocol, k, result["all_questions"], flush=True)
    return {key: value for key, value in result.items() if key != "per_question"}


def run_dataset(dataset, out, overlap):
    started = time.time()
    cv = load("30_run_loco_cv_selector.py")
    helper = load("108_run_v3_9_24_feature_card_quota_cv.py" if dataset == "locomo" else "120_run_longmemeval_24_feature_card_selector.py")
    summary_path = ROOT / ("outputs/eval/cv/v3_9_24_feature_card_guided_expand120/card_quota_24_feature_summary.json" if dataset == "locomo" else "outputs/eval/longmemeval_v3_10_expand120_24_feature_cardce_selector_top50summary_top50/summary.json")
    source = read(summary_path)
    paths = {k: Path(source[k]) for k in ("graph", "candidates", "cardce_paths")}
    params = dict(source.get("lgbm", {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 31,
                                     "min_child_samples": 10, "n_jobs": 8, "random_state": 13}))
    save(out / "manifest.json", {"source_summary": str(summary_path), "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
                                 "lgbm": params, "features": helper.RETRIEVAL_ROUTE_CARD5,
                                 "script_sha256": sha(__file__), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "note": "frozen development configuration, not nested CV; no QA/judge; same existing memory/candidates/card quota"})
    print(f"Loading {dataset} graph/candidates", flush=True)
    graph = JsonGraphStore().load(paths["graph"])
    items = read(paths["candidates"])
    if dataset == "locomo":
        items = [item for item in items if str(item.get("category")) in {"1", "2", "3", "4"}]
        assert len(items) == 1540
        conversations = cv.ordered_conversations(items)
        folds = [[str(item["question_id"]) for item in items if cv.conversation_id_from_question(item["question_id"]) == c] for c in conversations]
        ks = [20]
        originals = {20: ROOT / "outputs/paths/full_v3_9_card_guided_expand120_light_quota_top20.json"}
    else:
        items = helper.truncate_candidates(helper.canonicalize_longmemeval_candidates(items), source["candidate_topn"])
        assert len(items) == 500
        folds = [f["test_qids"] for f in overlap["grouped_folds"]] if overlap["grouped_cv_feasible"] else []
        ks = [20, 50]
        originals = {50: summary_path.parent / "card_quota_light_top50_paths.json"}
    qids = {str(item["question_id"]) for item in items}
    assert len(qids) == len(items)
    metrics = []
    original = read(originals[max(originals)])
    original = [item for item in original if str(item["question_id"]) in qids]
    assert {str(item["question_id"]) for item in original} == qids
    for k in ks:
        # For LME, @20 is the first 20 of the actual frozen Top50 QA order.
        metrics.append(evaluate(graph, original, k, "Original", out))
        metrics.append(evaluate(graph, ce_items(items, k), k, "Same_pool_CE", out))
    if not folds:
        save(out / "summary.json", {"status": "grouped_cv_blocked", "reason": "Insufficient independent history components; no forced split or replacement experiment", "metrics": metrics})
        return
    cache = cv.build_feature_cache(graph, items)
    card_index = helper.build_cardce_index(read(paths["cardce_paths"]))
    audit = {}
    for item in items:
        qid = str(item["question_id"])
        labels = cache[qid]["labels"]
        gold = [normalize_evidence_id(g) for g in item.get("gold_evidence", [])]
        conv = cv.conversation_id_from_question(qid)
        unresolved = [g for g in gold if f"{conv}:raw:{g}" not in graph.nodes]
        audit[qid] = {"category": item.get("category", item.get("question_type")), "gold_count": len(gold),
                      "unresolved_gold_ids": unresolved, "candidates": len(labels), "positives": sum(labels),
                      "negatives": len(labels) - sum(labels), "all_negative": bool(labels) and not any(labels)}
    save(out / "label_audit.json", audit)
    aggregate = {k: [] for k in ks}
    manifest = []
    seen = set()
    for i, test_qids in enumerate(folds):
        test = set(test_qids)
        assert not (test & seen)
        seen |= test
        train_qids = [str(item["question_id"]) for item in items if str(item["question_id"]) not in test and (dataset == "locomo" or helper.has_gold(item))]
        assert not set(train_qids) & test
        if dataset == "locomo":
            assert not {cv.conversation_id_from_question(q) for q in train_qids} & {cv.conversation_id_from_question(q) for q in test}
        rows, labels, sizes, kept = cv.flatten_feature_cache(cache, train_qids)
        assert sum(sizes) == len(labels) == len(rows)
        record = {"fold": i, "train_qids": kept, "test_qids": sorted(test), "train_rows": len(labels),
                  "positive_rows": sum(labels), "negative_rows": len(labels) - sum(labels), "random_state": params["random_state"] + i}
        manifest.append(record)
        save(out / "split_manifest.json", manifest)
        print(f"{dataset} fold={i} train={len(kept)} test={len(test)} rows={len(labels)}", flush=True)
        model = train_lightgbm_ranker(project_feature_rows(rows, helper.RETRIEVAL_ROUTE_CARD5), labels, sizes,
                                     **{**params, "random_state": params["random_state"] + i})
        model.booster_.save_model(str(out / f"fold_{i:02d}_model.txt"))
        test_items = [item for item in items if str(item["question_id"]) in test]
        scores = helper.score_items_with_cache(test_items, model, cache, helper.RETRIEVAL_ROUTE_CARD5) if dataset != "locomo" else cv.score_items_with_cache(test_items, model, cache, helper.RETRIEVAL_ROUTE_CARD5)
        for k in ks:
            selected = helper.card_quota_items(test_items, scores, card_index, topk=k, card_quota=5, method="Corrected")
            save(out / f"fold_{i:02d}_top{k}_paths.json", selected)
            aggregate[k].extend(selected)
    assert seen == qids
    for k in ks:
        metrics.append(evaluate(graph, sorted(aggregate[k], key=lambda x: x["question_id"]), k, "Corrected", out))
    save(out / "summary.json", {"status": "completed", "seconds": time.time() - started, "metrics": metrics})


def render(out):
    lines = ["# Frozen-candidate selector protocol audit", "", "CPU only; no generation or judge. This is a post-hoc protocol correction, not an untouched test evaluation.", ""]
    for dataset in ("locomo", "longmemeval"):
        path = out / dataset / "summary.json"
        if not path.exists():
            continue
        summary = read(path)
        lines += [f"## {dataset}: {summary['status']}", "", "| Protocol | K | All questions | With gold | Hit (%) | Recall (%) | FullCover (%) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for entry in summary["metrics"]:
            metric = entry["all_questions"] if dataset == "locomo" else entry["with_gold"]
            lines.append(f"| {entry['protocol']} | {entry['k']} | {entry['all_questions']['num_questions']} | {entry['with_gold']['num_questions']} | {100*metric['hit']:.4f} | {100*metric['recall']:.4f} | {100*metric['full_cover']:.4f} |")
        lines += ["", "LoCoMo uses all Cat1-4 questions; LongMemEval retrieval uses questions with gold evidence (all-question diagnostics also saved). LME @20 truncates the Top50 output order with the same quota=5.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/analysis/selector_protocol_correction_20260916")
    parser.add_argument("--stage", choices=["audit", "locomo", "longmemeval", "all"], default="all")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    overlap = history_audit(out) if args.stage in {"audit", "all"} else read(out / "history_overlap_audit.json")
    if args.stage == "audit":
        return
    for dataset in ("locomo", "longmemeval"):
        if args.stage in {dataset, "all"}:
            target = out / dataset
            target.mkdir(exist_ok=True)
            run_dataset(dataset, target, overlap)
            render(out)
    save(out / "completion.json", {"status": "finished_requested_stages", "stage": args.stage, "time": time.time()})


if __name__ == "__main__":
    main()
