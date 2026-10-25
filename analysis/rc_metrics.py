"""Pure risk--coverage computations extracted from the paper notebook.

The functions in this module consume caller-supplied rows. They do not load
experiment artifacts, execute models, or contain paper result arrays.

The ``historical_*`` names are intentional. In the paper-producing notebook,
rows with failed correctness or invalid counts are removed before the
hard-redaction (HR) baseline groups by ``(entity, atom_id)`` and takes
``DataFrameGroupBy.first``. If an original row was removed, a later abstraction
can therefore become the HR row. This module preserves that behavior for exact
historical reproduction; it does not claim that the fallback is a strict
index-zero baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ROW_COLUMNS = {
    "entity",
    "atom_id",
    "abstraction",
    "confidence",
    "broad_count",
    "specific_count",
    "correctness",
}


def information_coverage(specific_count: float, broad_count: float) -> float:
    """Return the notebook's per-claim information score before clipping.

    ``specific_count == 0`` is the historical operational fallback to zero.
    Valid paper rows have ``broad_count > 1`` and ``specific_count >= 0``;
    :func:`prepare_rows_historical` applies those filters and clips this score
    to ``[0, 1]``.
    """

    if specific_count == 0:
        return 0.0
    return float(1 - (np.log2(specific_count) / np.log2(broad_count)))


def _as_frame(rows: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def prepare_rows_historical(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply the exact notebook filtering and label conversion order.

    Input rows must already contain the historical ``broad_count`` used by the
    experiment, including any upstream broad-people override. Correctness must
    use the notebook labels ``SUPPORTED``, ``UNSUPPORTED``, or ``FAILED``.
    """

    df = _as_frame(rows)
    missing = sorted(REQUIRED_ROW_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    # Mechanically preserved from notebook cell 8, process_rows_to_df.
    df = df[df["correctness"] != "FAILED"]
    df = df[df["broad_count"] > 1]
    df = df[df["specific_count"] >= 0]
    df = df.copy()
    df["coverage"] = df.apply(
        lambda row: information_coverage(
            row["specific_count"], row["broad_count"]
        ),
        axis=1,
    )
    df["coverage"] = df["coverage"].clip(0, 1)
    df["correctness"] = df["correctness"].map(
        {"SUPPORTED": 1, "UNSUPPORTED": 0}
    )
    return df


def historical_sa_selection(prepared_rows: pd.DataFrame, theta: float) -> pd.DataFrame:
    """Select Aw-SA rows using the paper notebook's filter-before-group rule."""

    selected = prepared_rows[prepared_rows["confidence"] >= theta]
    selected = selected.groupby(["entity", "atom_id"]).first().reset_index()
    return selected.drop_duplicates(subset=["entity", "abstraction"])


def historical_hr_selection(prepared_rows: pd.DataFrame, theta: float) -> pd.DataFrame:
    """Select the historical HR baseline, including its documented fallback.

    Grouping occurs after invalid rows were filtered, and before the confidence
    threshold is applied. ``groupby(...).first()`` is retained verbatim because
    it is part of the paper-producing behavior.
    """

    baseline = prepared_rows.groupby(["entity", "atom_id"]).first().reset_index()
    baseline = baseline.drop_duplicates(subset=["entity", "abstraction"])
    return baseline[baseline["confidence"] >= theta]


def aggregate_selected_rows(selected_rows: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate selected rows exactly as the RC notebook does.

    Risk is claim-count weighted: ``1 - mean(correctness)``. It is not weighted
    by information coverage. Coverage is the sum of per-row information scores.
    Risk is returned in percentage units, matching the saved RC CSVs.
    """

    coverage = float(selected_rows["coverage"].sum())
    mean_correctness = float(selected_rows["correctness"].mean())
    return {
        "coverage": coverage,
        # ``float`` is equivalent to the notebook's scalar ``.item()`` while
        # also supporting pandas versions whose reduction already returns a
        # Python float.
        "risk": (1 - mean_correctness) * 100.0,
        "n_selected": int(len(selected_rows)),
    }


def build_historical_rc_curve(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the historical Aw-SA and HR curves from caller-supplied rows.

    The default threshold grid is the notebook's 101 points from 0 through 100
    inclusive. Raw coverage sums and percentage risks are returned; use one of
    the explicit normalization helpers for plotting coordinates.
    """

    prepared = prepare_rows_historical(rows)
    theta_values = (
        np.linspace(0, 100, 101)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )

    curve_rows: list[dict[str, float | int]] = []
    for theta in theta_values:
        sa = aggregate_selected_rows(historical_sa_selection(prepared, float(theta)))
        hr = aggregate_selected_rows(historical_hr_selection(prepared, float(theta)))
        curve_rows.append(
            {
                "theta": float(theta),
                "coverage_sa": sa["coverage"],
                "risk_sa": sa["risk"],
                "n_sa": sa["n_selected"],
                "coverage_hr": hr["coverage"],
                "risk_hr": hr["risk"],
                "n_hr": hr["n_selected"],
            }
        )
    return pd.DataFrame(curve_rows)


def normalize_single_plot_coverage_historical(curve: pd.DataFrame) -> pd.DataFrame:
    """Normalize both curves by maximum Aw-SA coverage, as notebook cell 7.

    Empty, non-finite, and non-positive normalization domains raise
    ``ValueError`` instead of producing non-finite coordinates.
    """

    result = curve.copy()
    if result.empty:
        raise ValueError("cannot normalize an empty risk--coverage curve")
    coverage_sa = result["coverage_sa"].to_numpy(dtype=float)
    coverage_hr = result["coverage_hr"].to_numpy(dtype=float)
    if not np.isfinite(coverage_sa).all() or not np.isfinite(coverage_hr).all():
        raise ValueError("coverage columns must contain only finite values")
    max_coverage = float(coverage_sa.max())
    if max_coverage <= 0:
        raise ValueError("Aw-SA maximum coverage must be greater than zero")
    result["coverage_sa"] = 100.0 * result["coverage_sa"] / max_coverage
    result["coverage_hr"] = 100.0 * result["coverage_hr"] / max_coverage
    return result


def normalize_grid_coverage_historical(curve: pd.DataFrame) -> pd.DataFrame:
    """Normalize by max(Aw-SA, HR) coverage, as the notebook's grid plot.

    Empty, non-finite, and non-positive normalization domains raise
    ``ValueError`` instead of producing non-finite coordinates.
    """

    result = curve.copy()
    if result.empty:
        raise ValueError("cannot normalize an empty risk--coverage curve")
    coverage_sa = result["coverage_sa"].to_numpy(dtype=float)
    coverage_hr = result["coverage_hr"].to_numpy(dtype=float)
    if not np.isfinite(coverage_sa).all() or not np.isfinite(coverage_hr).all():
        raise ValueError("coverage columns must contain only finite values")
    max_coverage = max(
        coverage_sa.max(),
        coverage_hr.max(),
    )
    if max_coverage <= 0:
        raise ValueError("maximum coverage must be greater than zero")
    result["coverage_sa"] = 100.0 * result["coverage_sa"] / max_coverage
    result["coverage_hr"] = 100.0 * result["coverage_hr"] / max_coverage
    return result


def risk_at_coverage_historical(
    coverage_sa: Sequence[float] | np.ndarray,
    risk_sa: Sequence[float] | np.ndarray,
    target_coverage: float,
) -> float:
    """Linearly interpolate Aw-SA risk at a target coverage.

    Risks at duplicate coverage values are averaged before ``numpy.interp``,
    matching notebook cell 19.
    """

    if target_coverage is None or np.isnan(target_coverage):
        return float("nan")
    curve = pd.DataFrame({"coverage": coverage_sa, "risk": risk_sa}).dropna()
    if curve.empty:
        return float("nan")
    curve = curve.sort_values("coverage")
    curve = curve.groupby("coverage", as_index=False)["risk"].mean()
    cov = curve["coverage"].to_numpy(dtype=float)
    risk = curve["risk"].to_numpy(dtype=float)
    if len(cov) == 0:
        return float("nan")
    return float(np.interp(float(target_coverage), cov, risk))


def aurc_full_range_historical(
    coverage: Sequence[float] | np.ndarray,
    risk: Sequence[float] | np.ndarray,
) -> float:
    """Compute the notebook's historical full-range AURC.

    This is the exact Cell-26 convention: drop NaNs; prepend coverage zero at
    the minimum-coverage risk; normalize coverage by this curve's own maximum;
    convert percentage risk to ``[0, 1]``; sort by coverage; and integrate with
    SciPy's trapezoid rule. The return value is in ``[0, 1]`` area units before
    the notebook's presentation-time multiplication by 100. Invalid or
    zero-width numeric domains raise ``ValueError``; valid-domain arithmetic
    is unchanged.
    """

    cov = np.asarray(coverage, dtype=float)
    r = np.asarray(risk, dtype=float)

    if cov.ndim != 1 or r.ndim != 1 or len(cov) != len(r):
        raise ValueError("coverage and risk must be one-dimensional and equal length")
    if np.isinf(cov).any() or np.isinf(r).any():
        raise ValueError("coverage and risk must not contain infinite values")
    mask = ~np.isnan(cov) & ~np.isnan(r)
    if mask.sum() == 0:
        raise ValueError("AURC requires at least one finite coverage--risk pair")
    cov = cov[mask]
    r = r[mask]

    min_idx = np.argmin(cov)
    min_cov = cov[min_idx]
    min_risk = r[min_idx]
    if min_cov > 0:
        cov = np.concatenate([[0.0], cov])
        r = np.concatenate([[min_risk], r])

    max_coverage = float(cov.max())
    if max_coverage <= 0:
        raise ValueError("AURC maximum coverage must be greater than zero")
    cov_norm = cov / max_coverage
    r_norm = r / 100.0
    order = np.argsort(cov_norm)
    cov_sorted = cov_norm[order]
    risk_sorted = r_norm[order]

    from scipy.integrate import trapezoid

    return float(trapezoid(y=risk_sorted, x=cov_sorted))


def aurc_shared_overlap(
    coverage: Sequence[float] | np.ndarray,
    risk: Sequence[float] | np.ndarray,
    lo: float,
    hi: float,
    max_coverage: float,
) -> float:
    """Compute shared-overlap AURC with the notebook's Cell-31 convention.

    Boundary risks are linearly interpolated. No coverage-zero anchor is added.
    Coverage is divided by the caller-supplied common denominator, and the area
    is not divided by the width of the overlap interval. Invalid or
    zero-width numeric domains raise ``ValueError``; valid-domain arithmetic
    is unchanged.
    """

    from scipy.integrate import trapezoid

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("shared AURC requires finite bounds with lo < hi")
    if not np.isfinite(max_coverage) or max_coverage <= 0:
        raise ValueError("shared AURC max_coverage must be finite and greater than zero")

    cov = np.asarray(coverage, dtype=float)
    r = np.asarray(risk, dtype=float)
    if cov.ndim != 1 or r.ndim != 1 or len(cov) != len(r):
        raise ValueError("coverage and risk must be one-dimensional and equal length")
    if np.isinf(cov).any() or np.isinf(r).any():
        raise ValueError("coverage and risk must not contain infinite values")
    mask = ~np.isnan(cov) & ~np.isnan(r)
    cov, r = cov[mask], r[mask]
    if len(cov) == 0:
        raise ValueError("shared AURC requires at least one finite coverage--risk pair")

    order = np.argsort(cov)
    cov, r = cov[order], r[order]
    if lo < cov[0] or hi > cov[-1]:
        raise ValueError("shared AURC bounds must lie within the observed coverage range")
    r_lo = float(np.interp(lo, cov, r)) if cov[0] <= lo <= cov[-1] else np.nan
    r_hi = float(np.interp(hi, cov, r)) if cov[0] <= hi <= cov[-1] else np.nan

    inner = (cov > lo) & (cov < hi)
    cov_clipped, risk_clipped = cov[inner], r[inner]
    if not np.isnan(r_lo):
        cov_clipped = np.concatenate([[lo], cov_clipped])
        risk_clipped = np.concatenate([[r_lo], risk_clipped])
    if not np.isnan(r_hi):
        cov_clipped = np.concatenate([cov_clipped, [hi]])
        risk_clipped = np.concatenate([risk_clipped, [r_hi]])
    if len(cov_clipped) < 2:
        raise ValueError("shared AURC requires at least two clipped curve points")

    coverage_normalized = cov_clipped / max_coverage
    risk_normalized = risk_clipped / 100.0
    return float(trapezoid(y=risk_normalized, x=coverage_normalized))


def shared_overlap_pair(
    coverage_a: Sequence[float] | np.ndarray,
    risk_a: Sequence[float] | np.ndarray,
    coverage_b: Sequence[float] | np.ndarray,
    risk_b: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute two AURCs on their shared range and relative improvement.

    ``improvement_percent`` treats curve B as the baseline, matching the paper
    notebook's ``100 * (baseline - candidate) / baseline`` convention.
    Empty, non-overlapping, or zero-baseline inputs raise ``ValueError``.
    """

    a = pd.DataFrame({"coverage": coverage_a, "risk": risk_a}).dropna()
    b = pd.DataFrame({"coverage": coverage_b, "risk": risk_b}).dropna()
    if a.empty or b.empty:
        raise ValueError("shared overlap requires two non-empty finite curves")

    cov_a = a["coverage"].to_numpy(dtype=float)
    r_a = a["risk"].to_numpy(dtype=float)
    cov_b = b["coverage"].to_numpy(dtype=float)
    r_b = b["risk"].to_numpy(dtype=float)
    lo = max(cov_a.min(), cov_b.min())
    hi = min(cov_a.max(), cov_b.max())
    max_coverage = max(cov_a.max(), cov_b.max())
    if lo >= hi:
        raise ValueError("curves must have a positive-width shared coverage range")
    if max_coverage <= 0 or not np.isfinite(max_coverage):
        raise ValueError("shared maximum coverage must be finite and greater than zero")
    aurc_a = aurc_shared_overlap(cov_a, r_a, lo, hi, max_coverage)
    aurc_b = aurc_shared_overlap(cov_b, r_b, lo, hi, max_coverage)

    aurc_a_x100 = aurc_a * 100.0
    aurc_b_x100 = aurc_b * 100.0
    if not np.isfinite(aurc_b_x100) or aurc_b_x100 == 0:
        raise ValueError("baseline AURC must be finite and non-zero for improvement")
    improvement = 100.0 * (aurc_b_x100 - aurc_a_x100) / aurc_b_x100
    return {
        "aurc_a_x100": float(aurc_a_x100),
        "aurc_b_x100": float(aurc_b_x100),
        "improvement_percent": float(improvement),
        "lo": float(lo),
        "hi": float(hi),
        "max_coverage": float(max_coverage),
    }
