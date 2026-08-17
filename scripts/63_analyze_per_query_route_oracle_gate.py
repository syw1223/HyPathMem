from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_path, write_json


DEFAULT_SETTINGS = {
    "EE": {
        "cv_dir": "outputs/eval/cv/graph_v3_2_gpt4o_eu_bottom_union_selector_lgbm_n80_top100",
        "variant": "route_agreement_v1",
    },
    "EH": {
        "cv_dir": "outputs/eval/cv/graph_v3_2_hyp_bottom_union_route_aware_v1_lgbm_n80_top100",
        "variant": "route_agreement_v1",
    },
    "E-EuHyp": {
        "cv_dir": "outputs/_archive_not_mainline_20260616/eval/cv/graph_v3_2_gpt4o_eu_hyp_union200_selector_lgbm_n80_top100",
        "variant": "retrieval_graph_v2_entity_session_topdown_route",
    },
    "EuHyp-H": {
        "cv_dir": "outputs/eval/cv/dual_geometry_B_true_bu_euhyp_td_hyp_lgbm_n80_top100",
        "variant": "route_agreement_v1",
    },
    "EuHyp-EuHyp": {
        "cv_dir": "outputs/eval/cv/dual_geometry_D_true_bu_euhyp_td_euhyp_lgbm_n80_top100",
        "variant": "route_aware_entity_v2",
    },
    "HH": {
        "cv_dir": "outputs/eval/cv/dual_geometry_HH_pure_bu_hyp_td_hyp_lgbm_n80_top100",
        "variant": "route_agreement_v2",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-json", default="outputs/eval/per_query_route_oracle_gate_best_variants.json")
    parser.add_argument("--output-md", default="outputs/eval/PER_QUERY_ROUTE_ORACLE_GATE_BEST_VARIANTS.md")
    args = parser.parse_args()

    setting_items = {
        name: load_cv_variant_eval(resolve_path(spec["cv_dir"]), spec["variant"])
        for name, spec in DEFAULT_SETTINGS.items()
    }
    question_ids = sorted(set.intersection(*(set(items) for items in setting_items.values())))
    if not question_ids:
        raise RuntimeError("no shared question ids across settings")

    per_setting_results = {}
    per_setting_summary = {}
    per_setting_utility = {}
    for setting, items in setting_items.items():
        results = {qid: items[qid] for qid in question_ids}
        per_setting_results[setting] = results
        per_setting_summary[setting] = summarize_metric_rows(list(results.values()))
        per_setting_utility[setting] = sum(utility(results[qid]) for qid in question_ids) / len(question_ids)

    gate_rows = []
    selected_results = []
    win_counts = {setting: 0 for setting in DEFAULT_SETTINGS}
    fractional_win_counts = {setting: 0.0 for setting in DEFAULT_SETTINGS}
    ee = per_setting_results["EE"]
    all_open = per_setting_results["EuHyp-EuHyp"]
    fixed_ee_utility = per_setting_utility["EE"]
    fixed_all_open_utility = per_setting_utility["EuHyp-EuHyp"]

    ee_miss_oracle_hit = 0
    ee_nonfull_oracle_full = 0
    all_open_miss_oracle_hit = 0
    all_open_nonfull_oracle_full = 0
    all_open_worse_utility_than_gate = 0
    all_open_same_utility_as_gate = 0

    for qid in question_ids:
        candidates = []
        for setting, results in per_setting_results.items():
            result = results[qid]
            candidates.append(
                {
                    "setting": setting,
                    "utility": utility(result),
                    "hit": float(result["hit"]),
                    "recall": float(result["recall"]),
                    "full_cover": float(result["full_cover"]),
                }
            )
        best_utility = max(row["utility"] for row in candidates)
        tied = [row for row in candidates if abs(row["utility"] - best_utility) < 1e-12]
        for row in tied:
            fractional_win_counts[row["setting"]] += 1.0 / len(tied)
        best = max(
            candidates,
            key=lambda row: (row["utility"], row["full_cover"], row["recall"], row["hit"], -list(DEFAULT_SETTINGS).index(row["setting"])),
        )
        win_counts[best["setting"]] += 1
        selected_results.append(per_setting_results[best["setting"]][qid])

        if not ee[qid]["hit"] and best["hit"]:
            ee_miss_oracle_hit += 1
        if not ee[qid]["full_cover"] and best["full_cover"]:
            ee_nonfull_oracle_full += 1
        if not all_open[qid]["hit"] and best["hit"]:
            all_open_miss_oracle_hit += 1
        if not all_open[qid]["full_cover"] and best["full_cover"]:
            all_open_nonfull_oracle_full += 1
        if utility(all_open[qid]) + 1e-12 < best["utility"]:
            all_open_worse_utility_than_gate += 1
        elif abs(utility(all_open[qid]) - best["utility"]) < 1e-12:
            all_open_same_utility_as_gate += 1

        gate_rows.append(
            {
                "question_id": qid,
                "winner": best["setting"],
                "winner_utility": best["utility"],
                "ties": [row["setting"] for row in tied],
                "settings": candidates,
            }
        )

    gate_summary = summarize_metric_rows(selected_results)
    gate_utility = sum(utility(result) for result in selected_results) / len(selected_results)
    payload = {
        "method": "per-query oracle route gate over final top-k outputs",
        "k": args.k,
        "utility_formula": "0.2*Hit + 0.35*Recall + 0.45*FullCover",
        "settings": DEFAULT_SETTINGS,
        "num_questions": len(question_ids),
        "per_setting_summary": per_setting_summary,
        "per_setting_utility": per_setting_utility,
        "oracle_gate_summary": gate_summary,
        "oracle_gate_utility": gate_utility,
        "win_counts": win_counts,
        "fractional_win_counts": fractional_win_counts,
        "improvements": {
            "vs_EE": {
                "delta_hit": gate_summary["hit"] - per_setting_summary["EE"]["hit"],
                "delta_recall": gate_summary["recall"] - per_setting_summary["EE"]["recall"],
                "delta_full_cover": gate_summary["full_cover"] - per_setting_summary["EE"]["full_cover"],
                "delta_utility": gate_utility - fixed_ee_utility,
                "miss_to_hit_questions": ee_miss_oracle_hit,
                "nonfull_to_full_questions": ee_nonfull_oracle_full,
            },
            "vs_EuHyp_EuHyp_fixed": {
                "delta_hit": gate_summary["hit"] - per_setting_summary["EuHyp-EuHyp"]["hit"],
                "delta_recall": gate_summary["recall"] - per_setting_summary["EuHyp-EuHyp"]["recall"],
                "delta_full_cover": gate_summary["full_cover"] - per_setting_summary["EuHyp-EuHyp"]["full_cover"],
                "delta_utility": gate_utility - fixed_all_open_utility,
                "miss_to_hit_questions": all_open_miss_oracle_hit,
                "nonfull_to_full_questions": all_open_nonfull_oracle_full,
                "higher_utility_questions": all_open_worse_utility_than_gate,
                "same_utility_questions": all_open_same_utility_as_gate,
            },
        },
        "per_query": gate_rows,
    }
    write_json(payload, resolve_path(args.output_json))
    write_markdown(payload, resolve_path(args.output_md))
    print(json.dumps({k: payload[k] for k in ["oracle_gate_summary", "oracle_gate_utility", "win_counts", "improvements"]}, indent=2))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")


def load_cv_variant_eval(cv_dir: Path, variant: str) -> dict[str, dict]:
    items = {}
    paths = sorted(cv_dir.glob(f"fold_*/*{variant}_eval.json"))
    if not paths:
        raise FileNotFoundError(f"no {variant}_eval.json under {cv_dir}")
    for path in paths:
        payload = read_json(path)
        for item in payload.get("per_question", []):
            qid = item["question_id"]
            if qid in items:
                raise ValueError(f"duplicate question_id {qid} from {path}")
            items[qid] = item
    return items


def utility(result) -> float:
    return 0.2 * float(result["hit"]) + 0.35 * float(result["recall"]) + 0.45 * float(result["full_cover"])


def summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "avg_tokens": 0.0,
            "avg_path_len": 0.0,
        }
    n = len(rows)
    return {
        "num_questions": n,
        "hit": sum(float(row["hit"]) for row in rows) / n,
        "recall": sum(float(row["recall"]) for row in rows) / n,
        "full_cover": sum(float(row["full_cover"]) for row in rows) / n,
        "avg_tokens": sum(float(row.get("tokens", 0.0)) for row in rows) / n,
        "avg_path_len": sum(float(row.get("path_len", 0.0)) for row in rows) / n,
    }


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown(payload: dict, path: Path) -> None:
    lines = []
    lines.append("# Per-Query Oracle Route Gate")
    lines.append("")
    lines.append("Utility: `U(q)=0.2*Hit + 0.35*Recall + 0.45*FullCover`.")
    lines.append("")
    lines.append("This analysis uses each setting's current best final top-5 selector output.")
    lines.append("")
    lines.append("## Fixed Setting Results")
    lines.append("")
    lines.append("| Setting | Variant | Hit@5 | Recall@5 | FullCover@5 | Utility |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for setting, spec in payload["settings"].items():
        summary = payload["per_setting_summary"][setting]
        lines.append(
            f"| {setting} | `{spec['variant']}` | {summary['hit']:.4f} | {summary['recall']:.4f} | "
            f"{summary['full_cover']:.4f} | {payload['per_setting_utility'][setting]:.4f} |"
        )
    lines.append("")
    lines.append("## Oracle Gate Result")
    lines.append("")
    summary = payload["oracle_gate_summary"]
    lines.append(
        f"- Oracle-gated final top-5: Hit={summary['hit']:.4f}, Recall={summary['recall']:.4f}, "
        f"FullCover={summary['full_cover']:.4f}, Utility={payload['oracle_gate_utility']:.4f}"
    )
    lines.append("")
    lines.append("## Winners")
    lines.append("")
    lines.append("| Setting | Deterministic Wins | Fractional Tie Wins |")
    lines.append("| --- | ---: | ---: |")
    for setting in payload["settings"]:
        lines.append(
            f"| {setting} | {payload['win_counts'][setting]} | {payload['fractional_win_counts'][setting]:.1f} |"
        )
    lines.append("")
    lines.append("## Improvement Space")
    lines.append("")
    for baseline, values in payload["improvements"].items():
        lines.append(f"### {baseline}")
        lines.append("")
        lines.append(f"- Delta Hit: {values['delta_hit']:.4f}")
        lines.append(f"- Delta Recall: {values['delta_recall']:.4f}")
        lines.append(f"- Delta FullCover: {values['delta_full_cover']:.4f}")
        lines.append(f"- Delta Utility: {values['delta_utility']:.4f}")
        lines.append(f"- Miss -> Hit questions: {values['miss_to_hit_questions']}")
        lines.append(f"- Non-full -> FullCover questions: {values['nonfull_to_full_questions']}")
        if "higher_utility_questions" in values:
            lines.append(f"- Higher-utility questions than fixed all-open: {values['higher_utility_questions']}")
            lines.append(f"- Same-utility questions as fixed all-open: {values['same_utility_questions']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
