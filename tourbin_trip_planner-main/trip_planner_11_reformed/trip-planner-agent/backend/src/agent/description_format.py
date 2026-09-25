"""Let the LLM choose readable headings while preserving graph facts verbatim."""

import json
import logging
import re
from functools import lru_cache

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.config import get_settings

logger = logging.getLogger(__name__)


def _segments(description: str) -> list[str]:
    """Split long database prose for layout without dropping any words."""
    text = " ".join(description.split())
    pieces = re.split(r"(?<=[.!؟؛])\s+", text)
    segments = []
    for piece in pieces:
        words = piece.split()
        for start in range(0, len(words), 35):
            segments.append(" ".join(words[start:start + 35]))
    return segments


def _assemble(sections: list, segments: list[str]) -> str | None:
    """Only accept ordered, complete coverage of every original text segment."""
    if not isinstance(sections, list) or not 1 <= len(sections) <= 3:
        return None
    rendered = []
    start = 0
    for section in sections:
        if not isinstance(section, dict):
            return None
        heading, end = section.get("heading"), section.get("end")
        if (not isinstance(heading, str) or not heading.strip() or len(heading) > 70
                or "\n" in heading or re.search(r"[0-9۰-۹]|کیلومتر|ساعت|دقیقه", heading)
                or type(end) is not int or end <= start or end > len(segments)):
            return None
        rendered.append(f"##### {heading.strip()}\n\n" + " ".join(segments[start:end]))
        start = end
    return "\n\n".join(rendered) if start == len(segments) else None


@lru_cache
def _formatter() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "You format Persian travel-destination descriptions for readability. "
            "Return ONLY a JSON object with key 'destinations', containing one object per input "
            "with the same 'name' and a 'sections' list. Each section has a short, descriptive "
            "Persian 'heading' (optionally one fitting emoji) and an integer 'end': the exclusive "
            "end index of the consecutive numbered text segments assigned to that section. "
            "For three segments, sections with end=1 and end=3 cover segment 0, then segments 1-2. "
            "Choose 1-3 meaningful headings per destination, covering all segments exactly once "
            "and in order. Do not write or repeat the descriptions, make new factual claims, "
            "or include travel distance/time in headings."
        ),
    )


async def format_descriptions(descriptions: dict[str, str]) -> dict[str, str]:
    """One LLM call for the entire set; on any failure keep all original text."""
    segments = {name: _segments(text) for name, text in descriptions.items() if text.strip()}
    if not segments:
        return {}
    fallback = {name: " ".join(parts) for name, parts in segments.items()}
    payload = {"destinations": [{"name": name, "segments": parts} for name, parts in segments.items()]}
    try:
        result = await _formatter().run(json.dumps(payload, ensure_ascii=False))
        output = result.output.strip()
        if output.startswith("```"):
            output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE)
        data = json.loads(output)
        choices = data.get("destinations")
        if not isinstance(choices, list):
            return fallback
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("name") not in segments:
                continue
            name = choice["name"]
            formatted = _assemble(choice.get("sections"), segments[name])
            if formatted:
                fallback[name] = formatted
    except Exception:
        logger.warning("Could not format destination descriptions; using original graph text", exc_info=True)
    return fallback
