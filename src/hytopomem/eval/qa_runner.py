from __future__ import annotations

from dataclasses import dataclass
import re

from hytopomem.eval.openai_compatible import ChatCompletionResult, ChatMessage, OpenAICompatibleChatClient


ANSWER_SYSTEM_PROMPT = (
    "You answer memory QA questions using only the provided context. "
    "Be concise and answer directly. Use explicit evidence and reasonable inference from the context, "
    "including dates, relative time, comparisons, preferences, and likely outcomes. "
    "Say you do not know only when the context contains no relevant evidence. "
    "Evidence timestamps are included when available; use them to resolve relative dates "
    "such as yesterday, last week, two days ago, recently, or this month."
)

ANSWER_USER_PROMPT = """Question:
{question}

Context:
{context}

Return only the answer, without extra explanation."""

ANSWER_V2_SYSTEM_PROMPT = (
    "You answer memory QA questions using only the provided evidence. "
    "Your job is to perform the correct answer operation for the question type, "
    "then return a short, direct final answer."
)

ANSWER_V2_USER_PROMPT = """Task-specific instruction:
{task_instruction}

General answer rules:
1. Give a short, direct answer. Do not add unrelated details.
2. Use the most specific answer supported by the evidence. If the evidence names a specific object, book, place, date, count, person, reason, or preference, use that exact detail rather than a broad category.
3. Do not answer "not specified" or "I don't know" if the evidence contains a direct or strongly supported answer.
4. If multiple evidence items conflict, prefer the evidence that directly matches the question's person, event, time, and object.
5. Do not mix in extra information that is not needed to answer the question.
6. Base the final answer only on the evidence, not on outside knowledge.

Question:
{question}

Evidence:
{context}

Final answer:"""

TASK_INSTRUCTIONS = {
    "count": (
        "This is a counting question. Count unique actual events/items only. "
        "Do not count plans, wishes, mentions, hypotheticals, or repeated descriptions of the same event. "
        "If two evidence items describe different occurrences, count both. "
        "Answer with the number and a very short explanation only if needed."
    ),
    "temporal": (
        "This is a temporal question. Use message timestamps carefully: a message timestamp is the time the message was sent, not always the event date. "
        "Resolve relative expressions such as yesterday, last week, tomorrow, next month, the week before, or recently relative to the message timestamp. "
        "Prefer the date/year/time window that directly matches the event asked in the question."
    ),
    "detail": (
        "This is a detail question. Return the most specific noun phrase supported by the evidence. "
        "Include key attributes such as object type, description, owner, color, name, place, or context when they are present. "
        "Do not answer with a broad category if the evidence gives a more specific item."
    ),
    "inference": (
        "This is a likely/would/why/how inference question. Infer the most likely answer from the evidence. "
        "Do not answer \"I don't know\" only because the future action or causal conclusion is not explicitly stated. "
        "If the evidence clearly supports a likely yes/no answer or reason, give that answer with a brief reason."
    ),
    "other": (
        "Answer directly from the evidence. Prefer the evidence that best matches the question's entity, event, time, and object."
    ),
}


class QARunner:
    def answer(self, question: str, context: str, category: int | None = None) -> str:
        raise NotImplementedError("Wire the answer LLM here after path retrieval is stable.")


@dataclass
class OpenAICompatibleQARunner(QARunner):
    client: OpenAICompatibleChatClient
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 128
    answer_protocol: str = "default"

    def answer(self, question: str, context: str, category: int | None = None) -> str:
        return self.answer_with_metadata(question, context, category=category).content.strip()

    def answer_with_metadata(
        self,
        question: str,
        context: str,
        category: int | None = None,
        question_type: str | None = None,
    ) -> ChatCompletionResult:
        result = self.client.chat_completion_with_metadata(
            model=self.model,
            messages=self._messages(question, context, category=category, question_type=question_type),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return ChatCompletionResult(
            content=result.content.strip(),
            usage=result.usage,
            elapsed_seconds=result.elapsed_seconds,
        )

    def _messages(
        self,
        question: str,
        context: str,
        category: int | None = None,
        question_type: str | None = None,
    ) -> list[ChatMessage]:
        qtype = question_type or classify_question(question)
        if self.answer_protocol == "v2_ops" and qtype in {"count", "temporal"}:
            task_instruction = TASK_INSTRUCTIONS.get(qtype, TASK_INSTRUCTIONS["other"])
            return [
                ChatMessage(role="system", content=ANSWER_V2_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=ANSWER_V2_USER_PROMPT.format(
                        task_instruction=task_instruction,
                        question=question,
                        context=context,
                    ),
                ),
            ]
        if self.answer_protocol == "v2":
            task_instruction = TASK_INSTRUCTIONS.get(qtype, TASK_INSTRUCTIONS["other"])
            return [
                ChatMessage(role="system", content=ANSWER_V2_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=ANSWER_V2_USER_PROMPT.format(
                        task_instruction=task_instruction,
                        question=question,
                        context=context,
                    ),
                ),
            ]
        return [
            ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=ANSWER_USER_PROMPT.format(question=question, context=context),
            ),
        ]


def classify_question(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(how many|number of|how much|times|months passed|years passed)\b", q):
        return "count"
    if re.search(r"\b(when|what date|which date|before|after|last week|next month|ago|year|month|day|weekend)\b", q):
        return "temporal"
    if re.search(r"\b(would|likely|might|could|if|considering|why|how does|how did)\b", q):
        return "inference"
    if re.search(r"\b(who|where|what kind|what type|what are the names|which|what book|what did|what is|what was)\b", q):
        return "detail"
    return "other"
