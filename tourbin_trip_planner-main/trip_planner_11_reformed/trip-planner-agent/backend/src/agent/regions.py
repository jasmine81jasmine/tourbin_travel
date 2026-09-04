"""Informal Iranian region-name expansion.

The graph only has real `City`/`Province` node names (`گیلان`, `آمل`,
`رامسر`, ...). Users very often ask in terms of an informal region instead
of a specific city/province -- "شمال" (the north / Caspian coast), "جنوب"
(the south), "کویر" (the desert belt), and so on. None of those words
appear as a City or Province name anywhere in the graph, so a plain
substring match against `location="شمال"` always returns zero rows, even
though the graph is full of Caspian-coast destinations (Gilan, Mazandaran,
Golestan).

`expand_location()` turns one such phrase into a list of concrete
city/province candidates that `TripGraphRepository.search_destinations`
can actually match against. It's intentionally a plain dict, not an LLM
call -- these regional groupings are stable geographic facts, not
something that needs to be inferred per-request.
"""

from __future__ import annotations

# alias -> concrete provinces/cities worth trying, in the graph's own vocabulary.
IRAN_REGION_ALIASES: dict[str, list[str]] = {
    "شمال": ["گیلان", "مازندران", "گلستان"],
    "شمال کشور": ["گیلان", "مازندران", "گلستان"],
    "شمال ایران": ["گیلان", "مازندران", "گلستان"],
    "ساحل خزر": ["گیلان", "مازندران", "گلستان"],
    "دریای خزر": ["گیلان", "مازندران", "گلستان"],
    "کرانه خزر": ["گیلان", "مازندران", "گلستان"],
    "جنوب": ["هرمزگان", "بوشهر", "خوزستان", "سیستان و بلوچستان", "فارس"],
    "جنوب کشور": ["هرمزگان", "بوشهر", "خوزستان", "سیستان و بلوچستان", "فارس"],
    "جنوب ایران": ["هرمزگان", "بوشهر", "خوزستان", "سیستان و بلوچستان", "فارس"],
    "غرب": ["کردستان", "کرمانشاه", "همدان", "ایلام", "آذربایجان غربی"],
    "غرب کشور": ["کردستان", "کرمانشاه", "همدان", "ایلام", "آذربایجان غربی"],
    "شمال غرب": ["آذربایجان شرقی", "آذربایجان غربی", "اردبیل", "زنجان"],
    "شمال غربی": ["آذربایجان شرقی", "آذربایجان غربی", "اردبیل", "زنجان"],
    "شرق": ["خراسان رضوی", "خراسان جنوبی", "خراسان شمالی", "سیستان و بلوچستان"],
    "شمال شرق": ["خراسان رضوی", "خراسان شمالی"],
    "مرکز": ["اصفهان", "یزد", "مرکزی", "قم", "سمنان"],
    "مرکز کشور": ["اصفهان", "یزد", "مرکزی", "قم", "سمنان"],
    "کویر": ["یزد", "کرمان", "سمنان", "اصفهان"],
    "اطراف تهران": ["تهران", "البرز", "قزوین"],
    "حومه تهران": ["تهران", "البرز", "قزوین"],
}


def expand_location(location: str | None) -> list[str]:
    """Turn one user-supplied location phrase into one or more concrete
    city/province candidates worth matching against the graph.

    - Exact/substring alias hit (e.g. "شمال", "می‌خوام برم شمال") -> the
      alias's province list.
    - No alias hit -> the original phrase, unchanged (still handled by the
      normal city/province substring match in the graph query).
    - Empty/None -> [].
    """
    if not location:
        return []
    text = location.strip()
    if not text:
        return []
    for alias, provinces in IRAN_REGION_ALIASES.items():
        if alias in text or text in alias:
            return provinces
    return [text]
