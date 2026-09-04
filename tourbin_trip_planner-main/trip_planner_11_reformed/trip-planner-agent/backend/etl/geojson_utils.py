"""Shared low-memory helpers for Overpass GeoJSON ETL commands."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson


PERSIAN_RE = re.compile(r"[\u0600-\u06ff]")
ARABIC_ORTHOGRAPHY_RE = re.compile(r"[\u0649\u064a\u0643\u0629\u0623\u0625\u0624\u0626]")


def has_persian(value: Any) -> bool:
    """Best-effort Persian detection for unlocalized OSM text.

    Arabic and Persian share a script. Arabic-only letter forms catch the
    common false positives while ambiguous strings remain usable.
    """
    return (
        isinstance(value, str)
        and bool(PERSIAN_RE.search(value))
        and not bool(ARABIC_ORTHOGRAPHY_RE.search(value))
    )


def persian_value(properties: dict[str, Any], *keys: str) -> str | None:
    """Return the first Persian-script value, never another localization."""
    for key in keys:
        value = properties.get(key)
        if isinstance(value, str) and (
            (key.endswith(":fa") and PERSIAN_RE.search(value)) or has_persian(value)
        ):
            return value.strip()
    return None


def iter_features(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one GeoJSON feature at a time without loading FeatureCollection."""
    with Path(path).open("rb") as source:
        yield from ijson.items(source, "features.item", use_float=True)
