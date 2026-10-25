"""Assemble evaluator JSON files into rows accepted by :mod:`analysis`.

This module reproduces the paper pipeline's SA/HR row assembly only.  It does
not discover files, run evaluators, filter rows, or compute risk--coverage
metrics.  Callers must provide the three JSON files explicitly.

The historical paper pipeline could replace the broad count for the exact
question ``"How many people are there?"``. That behavior is never implicit
here: ``broad_people_count_override`` is a required API argument and the CLI
requires ``--broad-people-count-override``. Pass ``none`` to retain the broad
counts stored in the input JSON, or provide the count universe explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from collections.abc import Sequence
from pathlib import Path
from typing import Any


BROAD_PEOPLE_QUESTION = "How many people are there?"


class RowAssemblyError(ValueError):
    """Raised when parallel evaluator outputs cannot be joined exactly."""


def load_json_array(path: str | Path) -> list[dict[str, Any]]:
    """Load one explicitly named JSON file and require a top-level array."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a top-level JSON array: {source}")
    return value


def _broad_count(
    abstraction_counts: dict[str, Any],
    broad_people_count_override: int | float | None,
) -> int | float:
    broad = abstraction_counts["counts"]["broad"]
    if (
        broad_people_count_override is not None
        and abstraction_counts.get("questions", {}).get("broad", "")
        == BROAD_PEOPLE_QUESTION
    ):
        return broad_people_count_override
    return broad


def assemble_rows(
    counts_data: list[dict[str, Any]],
    correctness_data: list[dict[str, Any]],
    confidence_data: list[dict[str, Any]],
    entity: str,
    *,
    broad_people_count_override: int | float | None,
) -> list[dict[str, Any]]:
    """Join one entity's evaluator outputs using the historical semantics.

    The historical runner used ``zip`` and broad exception handling, which
    could silently truncate unequal arrays or skip malformed triples. This
    release interface fails closed instead. Top-level and per-atom abstraction
    lengths must match, every triple must be well formed, and abstraction text
    must match exactly. An absent, null, or literal ``"FAILED"`` correctness
    result remains the historical label ``"FAILED"``.

    For abstraction index zero, atom confidence overrides abstraction
    confidence.  Later abstractions use their own confidence.
    """

    if broad_people_count_override is not None and (
        isinstance(broad_people_count_override, bool)
        or not isinstance(broad_people_count_override, numbers.Real)
        or not math.isfinite(float(broad_people_count_override))
        or broad_people_count_override <= 1
    ):
        raise RowAssemblyError(
            "broad_people_count_override must be none or a finite number greater than one"
        )

    top_counts = {
        "counts": len(counts_data),
        "correctness": len(correctness_data),
        "confidence": len(confidence_data),
    }
    if len(set(top_counts.values())) != 1:
        raise RowAssemblyError(
            "unequal top-level atom counts for "
            f"entity={entity!r}: counts={top_counts['counts']}, "
            f"correctness={top_counts['correctness']}, "
            f"confidence={top_counts['confidence']}"
        )

    rows: list[dict[str, Any]] = []
    for atom_position in range(len(counts_data)):
        atom_counts = counts_data[atom_position]
        atom_correctness = correctness_data[atom_position]
        atom_confidence = confidence_data[atom_position]
        try:
            count_abstractions = atom_counts["abstractions"]
            correctness_abstractions = atom_correctness["abstractions"]
            confidence_abstractions = atom_confidence["abstractions"]
            nested_counts = {
                "counts": len(count_abstractions),
                "correctness": len(correctness_abstractions),
                "confidence": len(confidence_abstractions),
            }
        except (KeyError, TypeError) as error:
            raise RowAssemblyError(
                f"malformed atom at entity={entity!r}, atom_position={atom_position}: "
                f"missing or invalid {error}"
            ) from error
        if len(set(nested_counts.values())) != 1:
            raise RowAssemblyError(
                "unequal abstraction counts at "
                f"entity={entity!r}, atom_position={atom_position}: "
                f"counts={nested_counts['counts']}, "
                f"correctness={nested_counts['correctness']}, "
                f"confidence={nested_counts['confidence']}"
            )

        for abstraction_position in range(nested_counts["counts"]):
            abstraction_counts = count_abstractions[abstraction_position]
            abstraction_correctness = correctness_abstractions[abstraction_position]
            abstraction_confidence = confidence_abstractions[abstraction_position]
            try:
                abstraction = abstraction_counts["abstraction"]
                if not (
                    abstraction
                    == abstraction_correctness["abstraction"]
                    == abstraction_confidence["abstraction"]
                ):
                    raise RowAssemblyError(
                        "abstraction text mismatch at "
                        f"entity={entity!r}, atom_position={atom_position}, "
                        f"abstraction_position={abstraction_position}"
                    )

                correctness_result = abstraction_correctness.get(
                    "correctness_result"
                )
                if not correctness_result or correctness_result == "FAILED":
                    correctness = "FAILED"
                elif isinstance(correctness_result, dict):
                    correctness = correctness_result["annotation"]
                else:
                    raise TypeError("correctness_result must be a mapping or FAILED")
                confidence = (
                    abstraction_confidence["confidence"]
                    if abstraction_confidence["index"] != 0
                    else atom_confidence["confidence"]
                )
                rows.append(
                    {
                        "entity": entity,
                        "atom_id": atom_counts["index"],
                        "atom": atom_counts["atom"],
                        "abstraction": abstraction,
                        "abstraction_id": abstraction_counts["index"],
                        "confidence": confidence,
                        "broad_count": _broad_count(
                            abstraction_counts, broad_people_count_override
                        ),
                        "specific_count": abstraction_counts["counts"]["specific"],
                        "correctness": correctness,
                    }
                )
            except RowAssemblyError:
                raise
            except (KeyError, TypeError) as error:
                raise RowAssemblyError(
                    "malformed abstraction triple at "
                    f"entity={entity!r}, atom_position={atom_position}, "
                    f"abstraction_position={abstraction_position}: {error}"
                ) from error
    return rows


def assemble_rows_from_files(
    counts_path: str | Path,
    correctness_path: str | Path,
    confidence_path: str | Path,
    entity: str,
    *,
    broad_people_count_override: int | float | None,
) -> list[dict[str, Any]]:
    """Load three explicit paths and assemble one entity's metric rows."""

    return assemble_rows(
        load_json_array(counts_path),
        load_json_array(correctness_path),
        load_json_array(confidence_path),
        entity,
        broad_people_count_override=broad_people_count_override,
    )


def _parse_override(value: str) -> int | float | None:
    if value.lower() == "none":
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected 'none' or a numeric broad-people count"
        ) from error
    if not math.isfinite(number) or number <= 1:
        raise argparse.ArgumentTypeError(
            "broad-people count must be finite and greater than one"
        )
    return int(number) if number.is_integer() else number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble explicit evaluator JSON files into metric rows."
    )
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--confidence", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--broad-people-count-override",
        required=True,
        type=_parse_override,
        metavar="COUNT|none",
        help=(
            "provide the intended count universe explicitly, or use 'none' "
            "to retain stored broad counts"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = assemble_rows_from_files(
        args.counts,
        args.correctness,
        args.confidence,
        args.entity,
        broad_people_count_override=args.broad_people_count_override,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
