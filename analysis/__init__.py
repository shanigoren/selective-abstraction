"""Code-only risk--coverage analysis for Selective Abstraction."""

from .rc_metrics import (
    aggregate_selected_rows,
    aurc_full_range_historical,
    aurc_shared_overlap,
    build_historical_rc_curve,
    historical_hr_selection,
    historical_sa_selection,
    information_coverage,
    normalize_grid_coverage_historical,
    normalize_single_plot_coverage_historical,
    prepare_rows_historical,
    risk_at_coverage_historical,
    shared_overlap_pair,
)

__all__ = [
    "aggregate_selected_rows",
    "aurc_full_range_historical",
    "aurc_shared_overlap",
    "build_historical_rc_curve",
    "historical_hr_selection",
    "historical_sa_selection",
    "information_coverage",
    "normalize_grid_coverage_historical",
    "normalize_single_plot_coverage_historical",
    "prepare_rows_historical",
    "risk_at_coverage_historical",
    "shared_overlap_pair",
]
