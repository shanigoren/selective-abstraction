"""Offline software-correctness tests for the public RC/AURC API.

All rows and curves in this file are invented. These tests do not constitute
research validation or reproduce any reported experiment.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import (
    aggregate_selected_rows,
    aurc_full_range_historical,
    aurc_shared_overlap,
    build_historical_rc_curve,
    historical_hr_selection,
    historical_sa_selection,
    normalize_grid_coverage_historical,
    normalize_single_plot_coverage_historical,
    prepare_rows_historical,
    shared_overlap_pair,
)
from analysis.assemble_rows import (
    RowAssemblyError,
    assemble_rows,
    assemble_rows_from_files,
    main as assemble_main,
)


def row(
    *,
    entity: str,
    atom_id: int,
    abstraction: str,
    abstraction_id: int,
    confidence: float,
    correctness: str = "SUPPORTED",
    broad_count: float = 16.0,
    specific_count: float = 4.0,
) -> dict[str, object]:
    return {
        "entity": entity,
        "atom_id": atom_id,
        "abstraction": abstraction,
        "abstraction_id": abstraction_id,
        "confidence": confidence,
        "broad_count": broad_count,
        "specific_count": specific_count,
        "correctness": correctness,
    }


class HistoricalSelectionTests(unittest.TestCase):
    def test_sa_selects_first_stored_abstraction_at_or_above_threshold(self) -> None:
        prepared = prepare_rows_historical(
            [
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="below",
                    abstraction_id=0,
                    confidence=69,
                ),
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="equal",
                    abstraction_id=1,
                    confidence=70,
                ),
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="later",
                    abstraction_id=2,
                    confidence=95,
                ),
            ]
        )

        selected = historical_sa_selection(prepared, theta=70)

        self.assertEqual(selected["abstraction"].tolist(), ["equal"])

    def test_exact_entity_abstraction_duplicates_are_removed(self) -> None:
        prepared = prepare_rows_historical(
            [
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="same text",
                    abstraction_id=0,
                    confidence=80,
                ),
                row(
                    entity="entity-a",
                    atom_id=1,
                    abstraction="same text",
                    abstraction_id=0,
                    confidence=80,
                ),
            ]
        )

        selected = historical_sa_selection(prepared, theta=80)

        self.assertEqual(len(selected), 1)

    def test_failed_original_row_preserves_historical_hr_fallback(self) -> None:
        prepared = prepare_rows_historical(
            [
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="failed original",
                    abstraction_id=0,
                    confidence=100,
                    correctness="FAILED",
                ),
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="fallback abstraction",
                    abstraction_id=1,
                    confidence=60,
                ),
            ]
        )

        selected = historical_hr_selection(prepared, theta=60)

        self.assertEqual(selected["abstraction"].tolist(), ["fallback abstraction"])

    def test_no_selected_rows_has_zero_coverage_and_nan_risk(self) -> None:
        rows = [
            row(
                entity="entity-a",
                atom_id=0,
                abstraction="low confidence",
                abstraction_id=0,
                confidence=10,
            )
        ]
        prepared = prepare_rows_historical(rows)
        selected = historical_sa_selection(prepared, theta=100)

        aggregate = aggregate_selected_rows(selected)
        curve = build_historical_rc_curve(rows, thresholds=[100])

        self.assertEqual(aggregate["coverage"], 0.0)
        self.assertEqual(aggregate["n_selected"], 0)
        self.assertTrue(math.isnan(aggregate["risk"]))
        self.assertEqual(curve.loc[0, "coverage_sa"], 0.0)
        self.assertTrue(math.isnan(curve.loc[0, "risk_sa"]))

    def test_zero_specific_count_maps_to_zero_information(self) -> None:
        prepared = prepare_rows_historical(
            [
                row(
                    entity="entity-a",
                    atom_id=0,
                    abstraction="zero count",
                    abstraction_id=0,
                    confidence=100,
                    specific_count=0,
                )
            ]
        )

        self.assertEqual(prepared["coverage"].tolist(), [0.0])


class AurcConventionTests(unittest.TestCase):
    def test_full_range_and_shared_overlap_remain_distinct(self) -> None:
        coverage = [20.0, 50.0, 100.0]
        risk = [10.0, 20.0, 40.0]

        full = aurc_full_range_historical(coverage, risk)
        shared = aurc_shared_overlap(
            coverage,
            risk,
            lo=20.0,
            hi=100.0,
            max_coverage=100.0,
        )

        self.assertAlmostEqual(full, 0.215)
        self.assertAlmostEqual(shared, 0.195)
        self.assertNotEqual(full, shared)

    def test_normalization_rejects_empty_and_all_zero_curves(self) -> None:
        empty = pd.DataFrame(columns=["coverage_sa", "coverage_hr"])
        all_zero = pd.DataFrame({"coverage_sa": [0.0], "coverage_hr": [0.0]})

        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_single_plot_coverage_historical(empty)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            normalize_single_plot_coverage_historical(all_zero)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            normalize_grid_coverage_historical(all_zero)

    def test_aurc_rejects_empty_and_zero_coverage_domains(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one finite"):
            aurc_full_range_historical([], [])
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            aurc_full_range_historical([0.0, 0.0], [10.0, 20.0])
        with self.assertRaisesRegex(ValueError, "max_coverage"):
            aurc_shared_overlap(
                [0.0, 1.0],
                [10.0, 20.0],
                lo=0.0,
                hi=1.0,
                max_coverage=0.0,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            shared_overlap_pair([], [], [], [])


def evaluator_documents() -> tuple[list[dict], list[dict], list[dict]]:
    """Return a small, wholly invented evaluator-output triple."""

    abstractions = [
        {
            "abstraction": "original claim",
            "index": 0,
            "counts": {"broad": 999, "specific": 10},
            "questions": {"broad": "How many people are there?"},
        },
        {
            "abstraction": "broader claim",
            "index": 1,
            "counts": {"broad": 20, "specific": 5},
        },
        {
            "abstraction": "unlabeled claim",
            "index": 2,
            "counts": {"broad": 30, "specific": 3},
        },
    ]
    counts = [{"index": 7, "atom": "invented atom", "abstractions": abstractions}]
    correctness = [
        {
            "abstractions": [
                {
                    "abstraction": "original claim",
                    "correctness_result": {"annotation": "SUPPORTED"},
                },
                {
                    "abstraction": "broader claim",
                    "correctness_result": {"annotation": "UNSUPPORTED"},
                },
                {"abstraction": "unlabeled claim"},
            ]
        }
    ]
    confidence = [
        {
            "confidence": 91,
            "abstractions": [
                {"abstraction": "original claim", "index": 0, "confidence": 12},
                {"abstraction": "broader claim", "index": 1, "confidence": 73},
                {"abstraction": "unlabeled claim", "index": 2, "confidence": 55},
            ],
        }
    ]
    return counts, correctness, confidence


class RowAssemblyTests(unittest.TestCase):
    def test_join_override_confidence_and_missing_label_semantics(self) -> None:
        counts, correctness, confidence = evaluator_documents()

        rows = assemble_rows(
            counts,
            correctness,
            confidence,
            "entity-invented",
            broad_people_count_override=12_345,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual([item["atom_id"] for item in rows], [7, 7, 7])
        self.assertEqual([item["confidence"] for item in rows], [91, 73, 55])
        self.assertEqual(
            [item["correctness"] for item in rows],
            ["SUPPORTED", "UNSUPPORTED", "FAILED"],
        )
        self.assertEqual(rows[0]["broad_count"], 12_345)

    def test_none_override_retains_stored_broad_count(self) -> None:
        counts, correctness, confidence = evaluator_documents()

        rows = assemble_rows(
            counts,
            correctness,
            confidence,
            "entity-invented",
            broad_people_count_override=None,
        )

        self.assertEqual(rows[0]["broad_count"], 999)

    def test_invalid_override_fails_closed(self) -> None:
        counts, correctness, confidence = evaluator_documents()

        with self.assertRaisesRegex(RowAssemblyError, "finite number"):
            assemble_rows(
                counts,
                correctness,
                confidence,
                "entity-invented",
                broad_people_count_override=float("nan"),
            )

    def test_explicit_json_files_and_cli_output(self) -> None:
        counts, correctness, confidence = evaluator_documents()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "counts": root / "counts.json",
                "correctness": root / "correctness.json",
                "confidence": root / "confidence.json",
            }
            for name, document in zip(
                ("counts", "correctness", "confidence"),
                (counts, correctness, confidence),
            ):
                paths[name].write_text(json.dumps(document), encoding="utf-8")

            loaded_rows = assemble_rows_from_files(
                paths["counts"],
                paths["correctness"],
                paths["confidence"],
                "entity-invented",
                broad_people_count_override=None,
            )
            output = root / "rows.json"
            exit_code = assemble_main(
                [
                    "--counts",
                    str(paths["counts"]),
                    "--correctness",
                    str(paths["correctness"]),
                    "--confidence",
                    str(paths["confidence"]),
                    "--entity",
                    "entity-invented",
                    "--output",
                    str(output),
                    "--broad-people-count-override",
                    "none",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), loaded_rows)

    def test_unequal_top_level_arrays_fail_with_counts(self) -> None:
        counts, correctness, confidence = evaluator_documents()
        confidence.clear()

        with self.assertRaisesRegex(
            RowAssemblyError,
            "counts=1, correctness=1, confidence=0",
        ):
            assemble_rows(
                counts,
                correctness,
                confidence,
                "entity-invented",
                broad_people_count_override=None,
            )

    def test_unequal_nested_arrays_fail_with_counts(self) -> None:
        counts, correctness, confidence = evaluator_documents()
        confidence[0]["abstractions"].pop()

        with self.assertRaisesRegex(
            RowAssemblyError,
            "counts=3, correctness=3, confidence=2",
        ):
            assemble_rows(
                counts,
                correctness,
                confidence,
                "entity-invented",
                broad_people_count_override=None,
            )

    def test_mismatched_abstraction_text_fails_closed(self) -> None:
        counts, correctness, confidence = evaluator_documents()
        correctness[0]["abstractions"][1]["abstraction"] = "different text"

        with self.assertRaisesRegex(RowAssemblyError, "abstraction_position=1"):
            assemble_rows(
                counts,
                correctness,
                confidence,
                "entity-invented",
                broad_people_count_override=None,
            )

    def test_malformed_triple_fails_closed(self) -> None:
        counts, correctness, confidence = evaluator_documents()
        malformed_counts = copy.deepcopy(counts)
        del malformed_counts[0]["abstractions"][0]["counts"]

        with self.assertRaisesRegex(RowAssemblyError, "malformed abstraction triple"):
            assemble_rows(
                malformed_counts,
                correctness,
                confidence,
                "entity-invented",
                broad_people_count_override=None,
            )


class CliIntegrationTests(unittest.TestCase):
    def test_json_assembly_to_curves_and_both_aurc_outputs(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assembled_paths = []
            for label, entity, atom_confidence, original_annotation in (
                ("a", "entity-a", 91, "UNSUPPORTED"),
                ("b", "entity-b", 60, "SUPPORTED"),
            ):
                counts, correctness, confidence = evaluator_documents()
                correctness[0]["abstractions"][0]["correctness_result"][
                    "annotation"
                ] = original_annotation
                confidence[0]["confidence"] = atom_confidence
                inputs = {}
                for name, document in zip(
                    ("counts", "correctness", "confidence"),
                    (counts, correctness, confidence),
                ):
                    path = root / f"{label}-{name}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    inputs[name] = path
                assembled = root / f"{label}-rows.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "analysis.assemble_rows",
                        "--counts",
                        str(inputs["counts"]),
                        "--correctness",
                        str(inputs["correctness"]),
                        "--confidence",
                        str(inputs["confidence"]),
                        "--entity",
                        entity,
                        "--output",
                        str(assembled),
                        "--broad-people-count-override",
                        "none",
                    ],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                assembled_paths.append(assembled)

            curves = root / "curves.csv"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "analysis",
                    "curves",
                    str(assembled_paths[0]),
                    str(curves),
                    "--additional-rows",
                    str(assembled_paths[1]),
                    "--thresholds",
                    "0,70,95",
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )

            full_output = root / "full.json"
            shared_output = root / "shared.json"
            for command in (
                [
                    "aurc-full",
                    str(curves),
                    str(full_output),
                    "--curve",
                    "sa",
                ],
                [
                    "aurc-shared",
                    str(curves),
                    str(shared_output),
                    "--curve-a",
                    "sa",
                    "--curve-b",
                    "hr",
                ],
            ):
                subprocess.run(
                    [sys.executable, "-B", "-m", "analysis", *command],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            curve_frame = pd.read_csv(curves)
            full_result = json.loads(full_output.read_text(encoding="utf-8"))
            shared_result = json.loads(shared_output.read_text(encoding="utf-8"))
            self.assertEqual(curve_frame["theta"].tolist(), [0.0, 70.0, 95.0])
            self.assertTrue(math.isfinite(full_result["aurc"]))
            self.assertEqual(full_result["variant"], "historical_full_range")
            self.assertEqual(shared_result["variant"], "shared_overlap")
            self.assertTrue(math.isfinite(shared_result["aurc_a_x100"]))
            self.assertTrue(math.isfinite(shared_result["aurc_b_x100"]))


if __name__ == "__main__":
    unittest.main()
