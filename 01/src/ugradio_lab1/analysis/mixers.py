"""DSB/SSB mixer analysis utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ugradio_lab1.utils.validation import as_1d_array


def expected_dsb_lines(
    f_lo_hz: float,
    f_rf_hz: float,
    *,
    orders: int = 3,
) -> pd.DataFrame:
    """Generate expected DSB/intermod line frequencies.

    ``orders`` controls the maximum harmonic order used in combinations
    ``|m*f_lo +/- n*f_rf|`` with ``1 <= m,n <= orders``.
    """

    f_lo = float(f_lo_hz)
    f_rf = float(f_rf_hz)
    if f_lo <= 0.0 or f_rf <= 0.0:
        raise ValueError("f_lo_hz and f_rf_hz must be positive.")
    if orders < 1:
        raise ValueError("orders must be >= 1.")

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        [
            {
                "family": "dsb_primary",
                "m_lo": 1,
                "n_rf": 1,
                "sign": "-",
                "expected_hz": abs(f_rf - f_lo),
                "formula": "|f_rf - f_lo|",
            },
            {
                "family": "dsb_primary",
                "m_lo": 1,
                "n_rf": 1,
                "sign": "+",
                "expected_hz": f_rf + f_lo,
                "formula": "f_rf + f_lo",
            },
        ]
    )

    for m in range(1, orders + 1):
        for n in range(1, orders + 1):
            plus = abs(m * f_lo + n * f_rf)
            minus = abs(m * f_lo - n * f_rf)
            rows.append(
                {
                    "family": "intermod",
                    "m_lo": m,
                    "n_rf": n,
                    "sign": "+",
                    "expected_hz": plus,
                    "formula": f"|{m}*f_lo + {n}*f_rf|",
                }
            )
            rows.append(
                {
                    "family": "intermod",
                    "m_lo": m,
                    "n_rf": n,
                    "sign": "-",
                    "expected_hz": minus,
                    "formula": f"|{m}*f_lo - {n}*f_rf|",
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame[frame["expected_hz"] > 0.0].copy()
    frame = frame.sort_values("expected_hz", kind="stable").drop_duplicates(
        subset=["expected_hz", "formula"], keep="first"
    )
    frame = frame.reset_index(drop=True)
    frame["line_id"] = [f"L{idx:03d}" for idx in range(frame.shape[0])]
    return frame[
        ["line_id", "family", "m_lo", "n_rf", "sign", "formula", "expected_hz"]
    ]


def match_observed_lines(
    expected_hz: pd.DataFrame | np.ndarray | Sequence[float],
    observed_hz: np.ndarray | Sequence[float],
    *,
    tolerance_hz: float,
) -> pd.DataFrame:
    """Match expected lines to nearest observed lines within a tolerance."""

    if tolerance_hz <= 0.0:
        raise ValueError("tolerance_hz must be positive.")

    expected_table, expected_values = _normalize_expected(expected_hz)
    observed = as_1d_array(observed_hz, "observed_hz", dtype=float)

    if observed.size == 0:
        raise ValueError("observed_hz cannot be empty.")

    used_observed: set[int] = set()
    matches: list[dict[str, float | int | bool | str]] = []

    for idx, row in expected_table.iterrows():
        expected_value = float(row["expected_hz"])
        deltas = np.abs(observed - expected_value)
        nearest_idx = int(np.argmin(deltas))
        nearest_delta = float(deltas[nearest_idx])
        within = nearest_delta <= tolerance_hz and nearest_idx not in used_observed
        if within:
            used_observed.add(nearest_idx)
            observed_value = float(observed[nearest_idx])
            observed_index = float(nearest_idx)
        else:
            observed_value = np.nan
            observed_index = np.nan

        matches.append(
            {
                **row.to_dict(),
                "observed_hz": observed_value,
                "observed_index": observed_index,
                "delta_hz": observed_value - expected_value if within else np.nan,
                "abs_delta_hz": nearest_delta if within else np.nan,
                "matched": bool(within),
                "match_tolerance_hz": float(tolerance_hz),
            }
        )

    # Include unmatched observed lines for completeness in spur surveys.
    for idx, observed_value in enumerate(observed):
        if idx in used_observed:
            continue
        matches.append(
            {
                "line_id": f"OBS{idx:03d}",
                "family": "unmatched_observed",
                "m_lo": np.nan,
                "n_rf": np.nan,
                "sign": "",
                "formula": "",
                "expected_hz": np.nan,
                "observed_hz": float(observed_value),
                "observed_index": float(idx),
                "delta_hz": np.nan,
                "abs_delta_hz": np.nan,
                "matched": False,
                "match_tolerance_hz": float(tolerance_hz),
            }
        )

    result = pd.DataFrame(matches)
    return result


def line_spur_catalog(
    *,
    config: str,
    f_lo_hz: float,
    f_rf_hz: float,
    observed_hz: np.ndarray | Sequence[float],
    observed_level_db: np.ndarray | Sequence[float] | None = None,
    tolerance_hz: float,
    orders: int = 3,
) -> pd.DataFrame:
    """Build a T7-style expected/observed line and spur catalog."""

    observed = as_1d_array(observed_hz, "observed_hz", dtype=float)
    if observed_level_db is not None:
        levels = as_1d_array(observed_level_db, "observed_level_db", dtype=float)
        if levels.size != observed.size:
            raise ValueError("observed_level_db must have the same length as observed_hz.")
    else:
        levels = np.full(observed.size, np.nan, dtype=float)

    expected = expected_dsb_lines(f_lo_hz=f_lo_hz, f_rf_hz=f_rf_hz, orders=orders)
    matched = match_observed_lines(expected, observed, tolerance_hz=tolerance_hz)

    level_by_index = {idx: float(level) for idx, level in enumerate(levels)}
    level_values: list[float] = []
    interpretation: list[str] = []
    for _, row in matched.iterrows():
        observed_idx = row.get("observed_index", np.nan)
        if np.isfinite(observed_idx):
            level_values.append(level_by_index.get(int(observed_idx), np.nan))
        else:
            level_values.append(np.nan)

        family = str(row["family"])
        is_matched = bool(row["matched"])
        if family == "dsb_primary" and is_matched:
            interpretation.append("expected_dsb_product")
        elif family == "intermod" and is_matched:
            interpretation.append("intermod_spur")
        elif family in {"dsb_primary", "intermod"} and not is_matched:
            interpretation.append("missing_expected_line")
        else:
            interpretation.append("unmatched_observed_spur")

    catalog = pd.DataFrame(
        {
            "config": np.full(matched.shape[0], config, dtype=object),
            "f_lo_hz": np.full(matched.shape[0], float(f_lo_hz), dtype=float),
            "f_rf_hz": np.full(matched.shape[0], float(f_rf_hz), dtype=float),
            "expected_line_hz": matched["expected_hz"].to_numpy(dtype=float),
            "observed_line_hz": matched["observed_hz"].to_numpy(dtype=float),
            "level_db": np.asarray(level_values, dtype=float),
            "interpretation": np.asarray(interpretation, dtype=object),
            "family": matched["family"].to_numpy(dtype=object),
            "formula": matched["formula"].to_numpy(dtype=object),
            "line_id": matched["line_id"].to_numpy(dtype=object),
        }
    )
    return catalog


def _normalize_expected(
    expected_hz: pd.DataFrame | np.ndarray | Sequence[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    if isinstance(expected_hz, pd.DataFrame):
        if "expected_hz" not in expected_hz.columns:
            raise ValueError("expected_hz DataFrame must include an 'expected_hz' column.")
        frame = expected_hz.copy()
        values = as_1d_array(frame["expected_hz"].to_numpy(dtype=float), "expected_hz", dtype=float)
        if "line_id" not in frame.columns:
            frame["line_id"] = [f"L{idx:03d}" for idx in range(frame.shape[0])]
        for col, default in [
            ("family", "expected"),
            ("m_lo", np.nan),
            ("n_rf", np.nan),
            ("sign", ""),
            ("formula", ""),
        ]:
            if col not in frame.columns:
                frame[col] = default
        frame = frame[["line_id", "family", "m_lo", "n_rf", "sign", "formula", "expected_hz"]]
        return frame, values

    values = as_1d_array(expected_hz, "expected_hz", dtype=float)
    frame = pd.DataFrame(
        {
            "line_id": [f"L{idx:03d}" for idx in range(values.size)],
            "family": ["expected"] * values.size,
            "m_lo": np.full(values.size, np.nan),
            "n_rf": np.full(values.size, np.nan),
            "sign": [""] * values.size,
            "formula": [""] * values.size,
            "expected_hz": values,
        }
    )
    return frame, values


__all__ = ["expected_dsb_lines", "line_spur_catalog", "match_observed_lines"]
