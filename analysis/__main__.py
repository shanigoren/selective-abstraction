"""Thin file-I/O interface for :mod:`analysis.rc_metrics`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rc_metrics import (
    aurc_full_range_historical,
    build_historical_rc_curve,
    shared_overlap_pair,
)


def _parse_thresholds(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--thresholds must contain at least one number")
    return np.asarray(values, dtype=float)


def _write_json(path: str, payload: dict[str, float | str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _curve_columns(name: str) -> tuple[str, str]:
    return f"coverage_{name}", f"risk_{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute code-only Selective Abstraction RC/AURC metrics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves = subparsers.add_parser(
        "curves", help="Build historical Aw-SA and HR curves from row CSV or JSON files."
    )
    curves.add_argument("input_csv")
    curves.add_argument("output_csv")
    curves.add_argument("--additional-rows", action="append", default=[],
                        help="Additional per-entity CSV or JSON row file; may be repeated.")
    curves.add_argument(
        "--thresholds",
        help="Comma-separated confidence thresholds; default is 0,...,100.",
    )

    full = subparsers.add_parser(
        "aurc-full", help="Compute the historical full-range AURC for one curve."
    )
    full.add_argument("input_csv")
    full.add_argument("output_json")
    full.add_argument("--curve", choices=("sa", "hr"), required=True)

    shared = subparsers.add_parser(
        "aurc-shared", help="Compare two curves on their shared coverage range."
    )
    shared.add_argument("input_csv")
    shared.add_argument("output_json")
    shared.add_argument("--curve-a", choices=("sa", "hr"), required=True)
    shared.add_argument("--curve-b", choices=("sa", "hr"), required=True)

    args = parser.parse_args()
    if args.command == "curves":
        frames = []
        for name in [args.input_csv, *args.additional_rows]:
            path = Path(name)
            if path.suffix.lower() == ".json":
                frames.append(pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))))
            else:
                frames.append(pd.read_csv(path))
        frame = pd.concat(frames, ignore_index=True)
        result = build_historical_rc_curve(frame, _parse_thresholds(args.thresholds))
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        return

    frame = pd.read_csv(args.input_csv)

    if args.command == "aurc-full":
        coverage_column, risk_column = _curve_columns(args.curve)
        value = aurc_full_range_historical(
            frame[coverage_column], frame[risk_column]
        )
        _write_json(
            args.output_json,
            {"variant": "historical_full_range", "curve": args.curve, "aurc": value},
        )
        return

    coverage_a, risk_a = _curve_columns(args.curve_a)
    coverage_b, risk_b = _curve_columns(args.curve_b)
    result = shared_overlap_pair(
        frame[coverage_a], frame[risk_a], frame[coverage_b], frame[risk_b]
    )
    _write_json(
        args.output_json,
        {
            "variant": "shared_overlap",
            "curve_a": args.curve_a,
            "curve_b": args.curve_b,
            **result,
        },
    )


if __name__ == "__main__":
    main()
