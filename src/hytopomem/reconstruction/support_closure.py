from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from hytopomem.reconstruction.schema import RawQuote


_TURN_RE = re.compile(r"^(?P<prefix>.+:t)(?P<turn>\d+)$")
_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those|they|them|he|she)\b", re.IGNORECASE)


class SupportClosureBuilder:
    """Recover compact, provenance-grounded raw support for a fact node."""

    def __init__(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        *,
        max_direct_chars: int = 3200,
        max_context_chars: int = 1800,
    ) -> None:
        self.nodes = nodes
        self.max_direct_chars = max_direct_chars
        self.max_context_chars = max_context_chars

    def build(self, node: Mapping[str, Any]) -> list[RawQuote]:
        metadata = node.get("metadata") or {}
        support_ids = list(metadata.get("support_raw_ids") or []) + list(node.get("support_ids") or [])
        if node.get("type") == "RAW" and node.get("node_id"):
            support_ids.insert(0, node["node_id"])

        quotes: list[RawQuote] = []
        seen: set[str] = set()
        for raw_id in support_ids:
            quote = self._quote(str(raw_id), support_kind="direct", max_chars=self.max_direct_chars)
            if quote is not None and quote.message_id not in seen:
                quotes.append(quote)
                seen.add(quote.message_id)

        direct_speaker = next((quote.speaker for quote in quotes if quote.speaker), str(metadata.get("speaker") or ""))
        needs_pair = direct_speaker.lower() == "assistant"
        needs_antecedent = bool(_PRONOUN_RE.search(str(node.get("text") or "")))
        if needs_pair or needs_antecedent:
            for quote in list(quotes):
                previous_id = previous_turn_id(quote.message_id)
                if not previous_id or previous_id in seen:
                    continue
                previous = self._quote(
                    previous_id,
                    support_kind="request_pair" if needs_pair else "antecedent_context",
                    max_chars=self.max_context_chars,
                )
                if previous is not None:
                    quotes.insert(0, previous)
                    seen.add(previous.message_id)
                    break
        return quotes

    def _quote(self, raw_id: str, *, support_kind: str, max_chars: int) -> RawQuote | None:
        raw = self.nodes.get(raw_id)
        if not raw or raw.get("type") != "RAW":
            return None
        metadata = raw.get("metadata") or {}
        text = str(raw.get("text") or "").strip()
        if not text:
            return None
        if len(text) > max_chars:
            text = text[: max_chars - 20].rstrip() + " [quote truncated]"
        return RawQuote(
            message_id=raw_id,
            text=text,
            speaker=str(metadata.get("speaker") or metadata.get("role") or ""),
            session_id=str(metadata.get("session_id") or metadata.get("session") or ""),
            message_time=str(raw.get("time") or metadata.get("timestamp") or "") or None,
            support_kind=support_kind,
        )


def previous_turn_id(raw_id: str) -> str | None:
    match = _TURN_RE.match(raw_id)
    if not match:
        return None
    turn_text = match.group("turn")
    turn = int(turn_text)
    if turn <= 0:
        return None
    return f"{match.group('prefix')}{turn - 1:0{len(turn_text)}d}"
