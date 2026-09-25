"""Short trip cards use graph facts, not unedited article prose or guessed ETA."""

import json
from types import SimpleNamespace

import pytest

from src.agent import description_format as formatting


def test_card_rejects_unverified_driving_figures():
    assert formatting._render_card({"why": "جنگل است", "activity": "پیاده‌روی",
                                    "facilities": "", "tip": "۲ ساعت رانندگی"}) is None
    assert formatting._render_card({"why": "طبیعت سرسبز", "activity": "پیاده‌روی",
                                    "facilities": "سرویس بهداشتی", "tip": "کفش مناسب"}) == (
        "- **چرا این مقصد؟** طبیعت سرسبز\n- **پیشنهاد بازدید** پیاده‌روی\n"
        "- **امکانات** سرویس بهداشتی\n- **نکتهٔ مسیر** کفش مناسب"
    )


@pytest.mark.asyncio
async def test_model_converts_graph_article_and_facilities_into_short_card(monkeypatch):
    article = "مقالهٔ مفصل و تکراری دربارهٔ درکه. " * 40

    class FakeAgent:
        async def run(self, prompt):
            payload = json.loads(prompt)
            assert payload["destinations"][0]["data"]["facilities"] == ["کافه", "رستوران"]
            return SimpleNamespace(output=json.dumps({"destinations": [{
                "name": "درکه", "why": "رودخانه و کوچه‌باغ‌های سرسبز دارد.",
                "activity": "صبح از مسیر پایین‌دست پیاده‌روی کنید.",
                "facilities": "کافه و رستوران دارد.", "tip": "کفش راحت همراه ببرید."
            }]}))

    monkeypatch.setattr(formatting, "_formatter", lambda: FakeAgent())
    cards = await formatting.format_descriptions({"درکه": {"description": article,
                                                            "facilities": ["کافه", "رستوران"]}})
    assert "**پیشنهاد بازدید**" in cards["درکه"]
    assert "**امکانات** کافه و رستوران دارد." in cards["درکه"]
    assert "مقالهٔ مفصل" not in cards["درکه"]


@pytest.mark.asyncio
async def test_bad_model_reply_falls_back_to_short_graph_fact_not_article(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"destinations":[]}')

    monkeypatch.setattr(formatting, "_formatter", lambda: FakeAgent())
    article = "در این مقاله با ما همراه باشید. " + "طبیعت زیبایی دارد. " * 40
    cards = await formatting.format_descriptions({"درکه": {"description": article, "categories": ["طبیعت"]}})
    assert "**چرا این مقصد؟**" in cards["درکه"]
    assert len(cards["درکه"]) < len(article)
