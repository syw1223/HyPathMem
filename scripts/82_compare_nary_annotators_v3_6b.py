from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json


ROLE_NAMES = (
    "old_state",
    "new_state",
    "preference_value",
    "polarity",
    "state_value",
    "plan_goal",
    "constraint",
    "temporal_scope",
    "reason_or_trigger",
    "reason_or_constraint",
    "exception",
    "context",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="outputs/nary_v3_6b/high_recall_annotations.json")
    parser.add_argument(
        "--candidate",
        default="outputs/nary_v3_6b/qwen30b_calibration_high_recall_maxtok800.json",
    )
    parser.add_argument("--reference-name", default="gpt-4o")
    parser.add_argument("--candidate-name", default="qwen30b")
    parser.add_argument("--output-json", default="outputs/eval/nary_v3_6b_qwen30b_vs_gpt4o_calibration.json")
    parser.add_argument("--output-md", default="outputs/eval/NARY_V3_6B_QWEN30B_VS_GPT4O_CALIBRATION.md")
    args = parser.parse_args()

    reference_payload = read_json(resolve_path(args.reference))
    candidate_payload = read_json(resolve_path(args.candidate))
    report = compare(reference_payload, candidate_payload, args.reference_name, args.candidate_name)
    report["reference_file"] = str(resolve_path(args.reference))
    report["candidate_file"] = str(resolve_path(args.candidate))

    out_json = resolve_path(args.output_json)
    write_json(report, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def compare(reference_payload: dict, candidate_payload: dict, reference_name: str, candidate_name: str) -> dict:
    ref = records_by_id(reference_payload)
    cand = records_by_id(candidate_payload)
    common_ids = sorted(set(ref) & set(cand))
    ref_only = sorted(set(ref) - set(cand))
    cand_only = sorted(set(cand) - set(ref))

    confusion = Counter()
    type_pairs = Counter()
    role_jaccards = []
    support_jaccards = []
    confidence_rows = []
    both_accept_type_match = 0
    both_accept = 0
    examples = defaultdict(list)

    for candidate_id in common_ids:
        r = annotation(ref[candidate_id])
        c = annotation(cand[candidate_id])
        r_accept = bool(r.get("accept"))
        c_accept = bool(c.get("accept"))
        confusion[(r_accept, c_accept)] += 1
        if len(examples[confusion_key(r_accept, c_accept)]) < 5:
            examples[confusion_key(r_accept, c_accept)].append(example_row(candidate_id, r, c))
        if r_accept and c_accept:
            both_accept += 1
            r_type = str(r.get("relation_type", ""))
            c_type = str(c.get("relation_type", ""))
            type_pairs[(r_type, c_type)] += 1
            if r_type == c_type:
                both_accept_type_match += 1
            role_jaccards.append(jaccard(filled_roles(r), filled_roles(c)))
            support_jaccards.append(jaccard(cited_facts(r), cited_facts(c)))
        confidence_rows.append(
            {
                "candidate_id": candidate_id,
                "reference_accept": r_accept,
                "candidate_accept": c_accept,
                "reference_confidence": safe_float(r.get("confidence")),
                "candidate_confidence": safe_float(c.get("confidence")),
            }
        )

    total = len(common_ids)
    agree = confusion[(False, False)] + confusion[(True, True)]
    summary = {
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_valid_ratio": valid_ratio(reference_payload),
        "candidate_valid_ratio": valid_ratio(candidate_payload),
        "reference_count": reference_payload.get("count", 0),
        "candidate_count": candidate_payload.get("count", 0),
        "common_count": total,
        "reference_only_count": len(ref_only),
        "candidate_only_count": len(cand_only),
        "reference_accept_rate": ratio(sum(bool(annotation(row).get("accept")) for row in ref.values()), len(ref)),
        "candidate_accept_rate": ratio(sum(bool(annotation(row).get("accept")) for row in cand.values()), len(cand)),
        "accept_agreement": ratio(agree, total),
        "accept_kappa": cohen_kappa(confusion),
        "both_accept_count": both_accept,
        "both_accept_type_agreement": ratio(both_accept_type_match, both_accept),
        "mean_role_jaccard_both_accept": mean(role_jaccards) if role_jaccards else 0.0,
        "mean_support_fact_jaccard_both_accept": mean(support_jaccards) if support_jaccards else 0.0,
    }
    confusion_table = {
        "reference_accept_candidate_accept": confusion[(True, True)],
        "reference_accept_candidate_reject": confusion[(True, False)],
        "reference_reject_candidate_accept": confusion[(False, True)],
        "reference_reject_candidate_reject": confusion[(False, False)],
    }
    return {
        "summary": summary,
        "confusion": confusion_table,
        "type_pairs_both_accept": {f"{a} -> {b}": n for (a, b), n in type_pairs.most_common()},
        "confidence_bins": confidence_bins(confidence_rows),
        "examples": dict(examples),
        "reference_only_ids": ref_only[:100],
        "candidate_only_ids": cand_only[:100],
    }


def records_by_id(payload: dict) -> dict[str, dict]:
    rows = {}
    for record in payload.get("records", []):
        candidate_id = str(record.get("candidate_id") or (record.get("candidate") or {}).get("candidate_id"))
        if candidate_id:
            rows[candidate_id] = record
    return rows


def annotation(record: dict) -> dict:
    value = record.get("annotation") or {}
    return value if isinstance(value, dict) else {}


def filled_roles(item: dict) -> set[str]:
    roles = item.get("roles") or {}
    out = set()
    for role_name in ROLE_NAMES:
        role = roles.get(role_name) or {}
        if str(role.get("value", "")).strip() and role.get("fact_ids"):
            out.add(role_name)
    return out


def cited_facts(item: dict) -> set[str]:
    roles = item.get("roles") or {}
    out = set()
    for role in roles.values():
        if isinstance(role, dict):
            out.update(str(fact_id) for fact_id in role.get("fact_ids", []))
    return out


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def valid_ratio(payload: dict) -> float:
    target = payload.get("target_count")
    if not target:
        return 1.0
    return ratio(payload.get("count", 0), target)


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def cohen_kappa(confusion: Counter) -> float:
    total = sum(confusion.values())
    if not total:
        return 0.0
    observed = (confusion[(True, True)] + confusion[(False, False)]) / total
    ref_true = (confusion[(True, True)] + confusion[(True, False)]) / total
    ref_false = 1.0 - ref_true
    cand_true = (confusion[(True, True)] + confusion[(False, True)]) / total
    cand_false = 1.0 - cand_true
    expected = ref_true * cand_true + ref_false * cand_false
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def confidence_bins(rows: list[dict]) -> list[dict]:
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.000001)]
    out = []
    for lo, hi in bins:
        bucket = [row for row in rows if lo <= row["candidate_confidence"] < hi]
        out.append(
            {
                "candidate_confidence_bin": f"[{lo:.1f},{min(hi, 1.0):.1f}]",
                "count": len(bucket),
                "candidate_accept_rate": ratio(sum(row["candidate_accept"] for row in bucket), len(bucket)),
                "reference_accept_rate": ratio(sum(row["reference_accept"] for row in bucket), len(bucket)),
                "accept_agreement": ratio(
                    sum(row["candidate_accept"] == row["reference_accept"] for row in bucket),
                    len(bucket),
                ),
                "mean_reference_confidence": mean([row["reference_confidence"] for row in bucket]) if bucket else 0.0,
            }
        )
    return out


def confusion_key(reference_accept: bool, candidate_accept: bool) -> str:
    if reference_accept and candidate_accept:
        return "both_accept"
    if reference_accept and not candidate_accept:
        return "reference_accept_candidate_reject"
    if not reference_accept and candidate_accept:
        return "reference_reject_candidate_accept"
    return "both_reject"


def example_row(candidate_id: str, reference: dict, candidate: dict) -> dict:
    return {
        "candidate_id": candidate_id,
        "reference_accept": bool(reference.get("accept")),
        "candidate_accept": bool(candidate.get("accept")),
        "reference_type": str(reference.get("relation_type", "")),
        "candidate_type": str(candidate.get("relation_type", "")),
        "reference_confidence": safe_float(reference.get("confidence")),
        "candidate_confidence": safe_float(candidate.get("confidence")),
        "reference_roles": sorted(filled_roles(reference)),
        "candidate_roles": sorted(filled_roles(candidate)),
        "support_fact_jaccard": jaccard(cited_facts(reference), cited_facts(candidate)),
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# N-ary V3.6B Annotator Calibration",
        "",
        f"Reference: `{s['reference']}`",
        f"Candidate: `{s['candidate']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "reference_valid_ratio",
        "candidate_valid_ratio",
        "reference_count",
        "candidate_count",
        "common_count",
        "reference_accept_rate",
        "candidate_accept_rate",
        "accept_agreement",
        "accept_kappa",
        "both_accept_count",
        "both_accept_type_agreement",
        "mean_role_jaccard_both_accept",
        "mean_support_fact_jaccard_both_accept",
    ]:
        lines.append(f"| {key} | {format_value(s[key])} |")
    lines.extend(["", "## Accept Confusion", "", "| Cell | Count |", "|---|---:|"])
    for key, value in report["confusion"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Candidate Confidence Bins", "", "| Qwen confidence | Count | Qwen accept | GPT accept | Agreement | GPT mean conf |", "|---|---:|---:|---:|---:|---:|"])
    for row in report["confidence_bins"]:
        lines.append(
            "| {candidate_confidence_bin} | {count} | {candidate_accept_rate:.4f} | "
            "{reference_accept_rate:.4f} | {accept_agreement:.4f} | {mean_reference_confidence:.4f} |".format(**row)
        )
    lines.extend(["", "## Type Pairs On Both-Accepted Items", "", "| Reference -> Candidate | Count |", "|---|---:|"])
    for pair, count in report["type_pairs_both_accept"].items():
        lines.append(f"| {pair} | {count} |")
    lines.extend(["", "## Example Disagreements", ""])
    for group in ["reference_reject_candidate_accept", "reference_accept_candidate_reject"]:
        lines.extend([f"### {group}", ""])
        for row in report["examples"].get(group, []):
            lines.append(f"- `{row['candidate_id']}` ref={row['reference_type']}({row['reference_confidence']:.2f}) cand={row['candidate_type']}({row['candidate_confidence']:.2f}) roles={row['candidate_roles']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
