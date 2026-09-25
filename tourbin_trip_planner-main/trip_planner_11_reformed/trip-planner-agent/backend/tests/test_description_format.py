"""LLM controls the layout, never the original destination facts."""

import json
from types import SimpleNamespace

import pytest

from src.agent import description_format as formatting


def test_invalid_llm_sections_cannot_omit_or_reorder_graph_content():
    segments = ["جنگل سرسبز است.", "مسیر دیدنی است."]
    assert formatting._assemble([{"heading": "🌿 طبیعت", "end": 1}], segments) is None
    assert formatting._assemble([{"heading": "🌿 طبیعت", "end": 2},
                                 {"heading": "مسیر", "end": 1}], segments) is None
    assert formatting._assemble([{"heading": "فاصله ۲ ساعت", "end": 2}], segments) is None
    assert formatting._assemble([{"heading": "🌿 طبیعت", "end": 1},
                                 {"heading": "مسیر", "end": 2}], segments) == (
        "##### 🌿 طبیعت\n\nجنگل سرسبز است.\n\n##### مسیر\n\nمسیر دیدنی است."
    )


@pytest.mark.asyncio
async def test_model_selects_headings_but_full_description_survives(monkeypatch):
    description = "جنگل سرسبز است. مسیر دیدنی است."

    class FakeAgent:
        async def run(self, prompt):
            assert json.loads(prompt)["destinations"][0]["segments"] == ["جنگل سرسبز است.", "مسیر دیدنی است."]
            return SimpleNamespace(output=json.dumps({"destinations": [{"name": "درکه", "sections": [
                {"heading": "🌿 طبیعت", "end": 1}, {"heading": "دسترسی", "end": 2}
            ]}]}))

    monkeypatch.setattr(formatting, "_formatter", lambda: FakeAgent())
    formatted = await formatting.format_descriptions({"درکه": description})
    assert formatted["درکه"] == "##### 🌿 طبیعت\n\nجنگل سرسبز است.\n\n##### دسترسی\n\nمسیر دیدنی است."


@pytest.mark.asyncio
async def test_bad_model_reply_uses_complete_graph_description(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"destinations":[{"name":"درکه","sections":[]}]}')

    monkeypatch.setattr(formatting, "_formatter", lambda: FakeAgent())
    assert await formatting.format_descriptions({"درکه": "یک متن کامل است."}) == {"درکه": "یک متن کامل است."}
