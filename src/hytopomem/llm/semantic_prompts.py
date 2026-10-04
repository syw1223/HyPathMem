from __future__ import annotations

import json
from typing import Sequence


PROMPT_VERSION = "graph_v3_2_semantic_json_v1"


EVENT_SYSTEM_PROMPT = """You are a memory compression module for a hierarchical retrieval system.

You are given a set of FACTS extracted from a conversation.

Your task:
- Group them into ONE event summary
- The event must represent a coherent local situation or activity
- Do NOT invent new facts
- Keep abstraction minimal but meaningful

Return STRICT JSON only:
{
  "event_title": "",
  "event_summary": "",
  "key_entities": [],
  "key_actions": [],
  "time_hint": ""
}

Rules:
- event_title: <= 12 words
- event_summary: 1 sentence only
- Only use information present in FACTS
- No hallucination
- If time is absent, set time_hint to an empty string
"""


TOPIC_SYSTEM_PROMPT = """You are a memory abstraction module.

You are given multiple EVENT summaries from a conversation.

Your task:
- Merge them into ONE high-level topic
- The topic should represent a general theme

Return STRICT JSON only:
{
  "topic_name": "",
  "topic_summary": "",
  "main_themes": [],
  "key_entities": []
}

Rules:
- topic_name: 2-6 words
- topic_summary: 1-2 sentences
- focus on abstraction, not detail
- no hallucination
"""


def event_user_prompt(facts: Sequence[dict]) -> str:
    return "FACTS:\n" + json.dumps(list(facts), ensure_ascii=False, indent=2)


def topic_user_prompt(events: Sequence[dict]) -> str:
    return "EVENTS:\n" + json.dumps(list(events), ensure_ascii=False, indent=2)


def event_text(annotation: dict) -> str:
    title = str(annotation.get("event_title") or "").strip()
    summary = str(annotation.get("event_summary") or "").strip()
    if title and summary:
        return f"{title}: {summary}"
    return summary or title or "Event summary unavailable"


def topic_text(annotation: dict) -> str:
    name = str(annotation.get("topic_name") or "").strip()
    summary = str(annotation.get("topic_summary") or "").strip()
    if name and summary:
        return f"{name}: {summary}"
    return summary or name or "Topic summary unavailable"

