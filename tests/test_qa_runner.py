from hytopomem.eval.qa_runner import OpenAICompatibleQARunner


class _UnusedClient:
    pass


def test_prompt_is_identical_across_categories() -> None:
    runner = OpenAICompatibleQARunner(client=_UnusedClient())

    category_two = runner._messages("When did it happen?", "Evidence", category=2)
    category_four = runner._messages("When did it happen?", "Evidence", category=4)

    assert category_two == category_four
    assert "Be concise and answer directly." in category_two[0].content
    assert "Return only the answer, without extra explanation." in category_two[1].content


def test_prompt_has_no_category_specific_format_rules() -> None:
    runner = OpenAICompatibleQARunner(client=_UnusedClient())

    messages = runner._messages("What was the profession?", "Evidence", category=4)

    combined = " ".join(message.content for message in messages)
    assert "under 10 words" not in combined
    assert "D Month YYYY" not in combined
    assert "Likely yes" not in combined
