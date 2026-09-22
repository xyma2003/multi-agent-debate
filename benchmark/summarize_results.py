#!/usr/bin/env python3
"""Summarize checked-in result files without calling a model.

The script deliberately separates raw runs, excludes validation sentinels from
paired type comparisons, and labels historical scores as key-factor mentions
rather than accuracy.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SENTINEL = "[Analysis unavailable due to validation error]"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values)


def print_base_summary() -> None:
    print("Base comparison")
    for name in ("full_system", "single_llm", "original_devil"):
        rows = _load(RESULTS / f"{name}.json")["results"]
        print(
            f"  {name:<18} n={len(rows):>2} "
            f"PDS={_mean(rows, 'pds'):.4f} "
            f"HR={_mean(rows, 'hedge_ratio'):.4f} "
            f"rounds={_mean(rows, 'rounds'):.2f}"
        )

    rows = _load(RESULTS / "nli_detection_fixed3rounds_baseline.json")["results"]
    multi_round = sum(row["rounds"] > 1 for row in rows)
    print(
        f"  {'nli canonical':<18} n={len(rows):>2} "
        f"PDS={_mean(rows, 'pds'):.4f} "
        f"HR={_mean(rows, 'hedge_ratio'):.4f} "
        f"SSS={_mean(rows, 'sss'):.4f} "
        f"rounds={_mean(rows, 'rounds'):.2f} "
        f"multi_round={multi_round}/{len(rows)}"
    )


def print_type_summary() -> None:
    data = _load(RESULTS / "type_comparison.json")
    bad_questions = {
        int(key.split("_", 1)[0])
        for key, positions in data["debates"].items()
        if any(SENTINEL in value for value in positions.values())
    }
    sentinel_positions = sum(
        SENTINEL in value
        for positions in data["debates"].values()
        for value in positions.values()
    )

    print("\nAdaptive comparison (matched valid questions only)")
    print(
        f"  excluded questions={sorted(bad_questions)}; "
        f"sentinel positions={sentinel_positions}"
    )
    for question_type in ("binary", "values_based", "context_dependent"):
        print(f"  {question_type}")
        for system in ("full_system", "adaptive_prohibition"):
            rows = [
                row
                for row in data["scores"]
                if row["question_type"] == question_type
                and row["system"] == system
                and row["question_id"] not in bad_questions
            ]
            focus = [value for row in rows for value in row["focus_dims"].values()]
            total = [row["scores"]["total"] for row in rows]
            print(
                f"    {system:<22} n={len(rows):>2} "
                f"focus={statistics.mean(focus):.3f} "
                f"total={statistics.mean(total):.3f}"
            )


def print_historical_summary() -> None:
    rows = _load(RESULTS / "historical" / "results.json")["scores"]
    print("\nHistorical key-factor evaluation (LLM-judged)")
    for system in ("full_system", "single_llm", "adaptive_prohibition"):
        scores = [row["score"] for row in rows if row["system"] == system]
        at_least_partial = sum(score >= 1 for score in scores)
        print(
            f"  {system:<22} n={len(scores):>2} "
            f"distribution={dict(sorted(Counter(scores).items()))} "
            f"partial_or_better={at_least_partial / len(scores):.0%}"
        )


def main() -> None:
    print_base_summary()
    print_type_summary()
    print_historical_summary()


if __name__ == "__main__":
    main()
