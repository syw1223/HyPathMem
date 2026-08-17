from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_path, write_json


VARIANTS = [
    (
        "H2 role-aware",
        "outputs/eval/cv/v4_0_hypercard_mp_batched_gpu7/hypercard_mp_summary.json",
        "h2_mp_lgbm",
        "Base44 + role-aware Fact-Card-Fact propagation",
    ),
    (
        "H2 membership-only",
        "outputs/eval/cv/v4_0_hypercard_mp_no_roles_gpu7/hypercard_mp_summary.json",
        "h2_mp_lgbm",
        "Base44 + membership propagation + card loss",
    ),
    (
        "H2 without card loss",
        "outputs/eval/cv/v4_0_hypercard_mp_no_card_loss_gpu7/hypercard_mp_summary.json",
        "h2_mp_lgbm",
        "Base44 + role-aware propagation, no card ranking loss",
    ),
    (
        "H3 hyperbolic attention",
        "outputs/eval/cv/v4_0_hyperbolic_hypercard_gpu7/hypercard_mp_summary.json",
        "h2_mp_lgbm",
        "Base44 + membership propagation + Lorentz distance attention",
    ),
    (
        "H2 mainline-aligned",
        "outputs/eval/cv/v4_0_h2_mainline_aligned_gpu7/hypercard_mp_summary.json",
        "h2_mp_lgbm",
        "Frozen Membership+Quality features + membership propagation + card loss",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/eval/V4_0_QUERY_INDUCED_HYPERGRAPH_SUMMARY.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/eval/V4_0_QUERY_INDUCED_HYPERGRAPH_SUMMARY.md",
    )
    args = parser.parse_args()

    frozen = load_json(
        resolve_path(
            "outputs/eval/cv/v3_9_clean_feature_search_top5/"
            "clean_feature_search_summary.json"
        )
    )["aggregate"]["clean_membership_quality"]
    rows = [
        {
            "variant": "H0 frozen Membership+Quality",
            "description": "Current frozen mainline; no message passing",
            "top5": metric_triplet(frozen),
            "top20": None,
        }
    ]
    for name, relative_path, method, description in VARIANTS:
        payload = load_json(resolve_path(relative_path))
        rows.append(
            {
                "variant": name,
                "description": description,
                "top5": metric_triplet(payload["aggregate"][method]["top5"]),
                "top20": metric_triplet(payload["aggregate"][method]["top20"]),
                "config": payload.get("config", {}),
            }
        )

    aligned_payload = load_json(
        resolve_path(
            "outputs/eval/cv/v4_0_h2_mainline_aligned_gpu7/"
            "hypercard_mp_summary.json"
        )
    )
    aligned_base = metric_triplet(aligned_payload["aggregate"]["base_lgbm"]["top5"])
    aligned_h2 = metric_triplet(aligned_payload["aggregate"]["h2_mp_lgbm"]["top5"])
    conclusion = {
        "keep_mainline": "H0 frozen Membership+Quality",
        "best_h2_ablation": "H2 membership-only",
        "mainline_aligned_delta": {
            key: aligned_h2[key] - aligned_base[key]
            for key in ("hit", "recall", "full_cover")
        },
        "decision": (
            "Keep H2/H3 as structural ablations. Mainline-aligned H2 does not "
            "produce a material gain over the frozen relation-card-aware ranker; "
            "H3 shifts slightly toward FullCover but reduces Hit/Recall."
        ),
    }
    output = {"rows": rows, "conclusion": conclusion}
    write_json(output, resolve_path(args.output_json))
    resolve_path(args.output_md).write_text(render_markdown(output), encoding="utf-8")
    print(render_markdown(output))


def metric_triplet(row: dict) -> dict:
    return {
        "hit": float(row["hit"]),
        "recall": float(row["recall"]),
        "full_cover": float(row["full_cover"]),
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_markdown(payload: dict) -> str:
    lines = [
        "# V4.0 Query-induced Hypergraph Reasoning",
        "",
        "| Variant | Hit@5 | Recall@5 | FullCover@5 | Hit@20 | Recall@20 | FullCover@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        top5 = row["top5"]
        top20 = row["top20"]
        top20_values = (
            [f"{top20[key]:.4f}" for key in ("hit", "recall", "full_cover")]
            if top20
            else ["-", "-", "-"]
        )
        lines.append(
            f"| {row['variant']} | {top5['hit']:.4f} | {top5['recall']:.4f} | "
            f"{top5['full_cover']:.4f} | {' | '.join(top20_values)} |"
        )
    delta = payload["conclusion"]["mainline_aligned_delta"]
    lines.extend(
        [
            "",
            "## Mainline-aligned H2 delta",
            "",
            f"- Hit@5: {delta['hit']:+.4f}",
            f"- Recall@5: {delta['recall']:+.4f}",
            f"- FullCover@5: {delta['full_cover']:+.4f}",
            "",
            "## Decision",
            "",
            payload["conclusion"]["decision"],
            "",
            "The role ablation indicates that noisy role labels are not helping. "
            "The card ranking loss is useful. Lorentz distance is more natural as "
            "an attention prior than as a flat feature, but it still does not yield "
            "a stable overall gain on this dataset.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
