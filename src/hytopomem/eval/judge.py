from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hytopomem.eval.openai_compatible import ChatMessage, OpenAICompatibleChatClient


ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc.

The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {answer}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class LLMJudge:
    def judge(self, question: str, answer: str, prediction: str) -> dict:
        raise NotImplementedError("Wire an LLM judge here for full QA evaluation.")


@dataclass
class OpenAICompatibleLLMJudge(LLMJudge):
    client: OpenAICompatibleChatClient
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 160

    def judge(self, question: str, answer: str, prediction: str) -> dict:
        judged = self.judge_with_metadata(question, answer, prediction)
        judged.pop("judge_usage", None)
        judged.pop("judge_elapsed_seconds", None)
        return judged

    def judge_with_metadata(self, question: str, answer: str, prediction: str) -> dict:
        result = self.client.chat_completion_with_metadata(
            model=self.model,
            messages=[
                ChatMessage(
                    role="user",
                    content=ACCURACY_PROMPT.format(
                        question=question,
                        answer=answer,
                        prediction=prediction,
                    ),
                )
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = result.content
        payload = parse_json_object(content)
        label = str(payload.get("label", "")).strip().upper()
        if label not in {"CORRECT", "WRONG"}:
            label = "WRONG"
        return {
            "judge_correct": 1 if label == "CORRECT" else 0,
            "judge_label": label,
            "judge_reason": str(payload.get("reason", "")).strip(),
            "judge_raw_response": content,
            "judge_usage": normalize_usage(result.usage),
            "judge_elapsed_seconds": result.elapsed_seconds,
        }


def parse_json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"label": "WRONG", "reason": f"judge returned non-JSON: {text[:120]}"}


def normalize_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
