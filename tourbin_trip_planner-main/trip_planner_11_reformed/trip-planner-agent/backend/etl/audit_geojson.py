"""Stream a GeoJSON file and report its complete property-key vocabulary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geojson_utils import has_persian, iter_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory audit of GeoJSON property keys")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", help="Output JSON path; defaults to stdout")
    parser.add_argument("--examples", type=int, default=3, help="Maximum distinct examples per key")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    persian_counts: Counter[str] = Counter()
    examples: dict[str, list[Any]] = defaultdict(list)
    feature_count = 0

    for feature in iter_features(args.path):
        feature_count += 1
        for key, value in (feature.get("properties") or {}).items():
            counts[key] += 1
            if has_persian(value):
                persian_counts[key] += 1
            if len(examples[key]) < args.examples and value not in examples[key]:
                examples[key].append(value)
        if args.progress_every and feature_count % args.progress_every == 0:
            print(f"Scanned {feature_count:,} features", file=sys.stderr)

    report = {
        "feature_count": feature_count,
        "unique_property_keys": len(counts),
        "properties": {
            key: {
                "count": counts[key],
                "persian_value_count": persian_counts[key],
                "examples": examples[key],
            }
            for key in sorted(counts, key=lambda item: (-counts[item], item))
        },
    }

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
            target.write("\n")
    else:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
