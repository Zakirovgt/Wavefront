from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CSVColumns:
    """
    Candidate CSV column names used to identify sensor derivatives, identifiers,
    coordinates, and activity flags.

    The loader selects the first matching name found in the input DataFrame.
    This makes the parsing code compatible with several common naming
    conventions.
    """

    zx_candidates: tuple[str, ...] = (
        "Zx",
        "Z_x",
        "zx",
        "ZX",
        "g1",
    )

    zy_candidates: tuple[str, ...] = (
        "Zy",
        "Z_y",
        "zy",
        "ZY",
        "g2",
    )

    n_candidates: tuple[str, ...] = (
        "N",
        "n",
        "sensor_id",
        "id",
    )

    x_candidates: tuple[str, ...] = (
        "X",
        "x",
    )

    y_candidates: tuple[str, ...] = (
        "Y",
        "y",
    )

    flag_candidates: tuple[str, ...] = (
        "Flag(0/1)",
        "flag",
    )


def pick_col(
        df: pd.DataFrame,
        candidates: Iterable[str],
        what: str,
) -> str:
    """
    Select the first available column name from a list of candidates.

    Args:
        df: Source DataFrame.
        candidates: Ordered sequence of acceptable column names.
        what: Human-readable description used in an error message.

    Returns:
        The first matching column name found in df.columns.

    Raises:
        ValueError: If none of the candidate names exists in the DataFrame.
    """
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Missing column for {what}. "
        f"Candidates={list(candidates)}; "
        f"got={list(df.columns)}"
    )


def to_deriv_array(
        meas: pd.DataFrame,
        ref: pd.DataFrame | None,
        align_by: AlignBy,
        flag_policy: FlagPolicy,
        cols: CSVColumns,
        xy_round: int = 6,
        dtype=np.float32,
) -> np.ndarray:
    """
    Convert measured sensor slopes into a standardized derivative array.

    The returned array stores two derivative components per sensor:

        output[:, 0] = dU/dx, corresponding to the Zx-like CSV column
        output[:, 1] = dU/dy, corresponding to the Zy-like CSV column

    When a reference DataFrame is supplied, measured values can be reordered
    to match the reference sensor layout using sensor identifiers or rounded
    x/y coordinates.

    Args:
        meas: DataFrame containing measured sensor slopes.
        ref: Optional reference DataFrame defining the desired sensor order.
            When None, the measured row order is used directly.
        align_by: Alignment strategy:
            - "row": Preserve the measured DataFrame row order.
            - "N": Align by a shared sensor identifier column.
            - "XY": Align by rounded x/y coordinates.
            - "auto": Prefer a shared identifier column; otherwise use x/y
              coordinate alignment.
        flag_policy: Policy for handling flagged sensor rows:
            - "zero": Keep all rows but set both derivative components to zero
              where the flag value is 0.
            - "drop": Remove rows where the flag value is 0.
        cols: Candidate-column configuration.
        xy_round: Decimal precision used when matching coordinates in "XY"
            alignment mode.
        dtype: NumPy dtype of the returned derivative array.

    Returns:
        Derivative array with shape (P_sensor, 2).

    Raises:
        ValueError: If required columns are missing, alignment leaves unmatched
            sensors, or the final array does not have shape (P_sensor, 2).

    Notes:
        The function does not modify the input DataFrames. A working copy of
        the measurement DataFrame is created before flag handling.
    """
    # Identify derivative columns using the configured naming conventions.
    zx = pick_col(meas, cols.zx_candidates, "Zx")
    zy = pick_col(meas, cols.zy_candidates, "Zy")

    # The flag column is optional.
    flag = next(
        (
            c
            for c in cols.flag_candidates
            if c in meas.columns
        ),
        None,
    )

    # Work on a copy so the caller's input DataFrame remains unchanged.
    df = meas.copy()

    # Apply the requested policy for inactive or invalid sensor rows.
    if flag and flag_policy == "zero":
        df.loc[df[flag] == 0, [zx, zy]] = 0.0

    elif flag and flag_policy == "drop":
        df = df[df[flag] == 1].copy()

    # Without a reference layout, or when explicitly requested, preserve the
    # current measurement row order.
    if ref is None or align_by == "row":
        out = df[[zx, zy]]
        return out.to_numpy(dtype=dtype)

    # In automatic mode, prefer an identifier-based merge when a compatible
    # identifier column is available in both tables; otherwise use coordinates.
    if align_by == "auto":
        n_col = next(
            (
                c
                for c in cols.n_candidates
                if c in df.columns and c in ref.columns
            ),
            None,
        )

        if n_col:
            align_by = "N"
        else:
            align_by = "XY"

    if align_by == "N":
        # Reorder slopes to follow the reference DataFrame's sensor-ID order.
        n = pick_col(df, cols.n_candidates, "N")

        merged = ref[[n]].merge(
            df[[n, zx, zy]],
            on=n,
            how="left",
            sort=False,
        )

        out = merged[[zx, zy]]

    else:  # "XY"
        # Match sensor rows by rounded Cartesian coordinates. Rounding helps
        # avoid failures caused by small floating-point representation errors.
        x = pick_col(df, cols.x_candidates, "X")
        y = pick_col(df, cols.y_candidates, "Y")

        xr = pick_col(ref, cols.x_candidates, "X")
        yr = pick_col(ref, cols.y_candidates, "Y")

        ref2 = ref.assign(
            _X=ref[xr].round(xy_round),
            _Y=ref[yr].round(xy_round),
        )

        df2 = df.assign(
            _X=df[x].round(xy_round),
            _Y=df[y].round(xy_round),
        )

        merged = ref2[["_X", "_Y"]].merge(
            df2[["_X", "_Y", zx, zy]],
            on=["_X", "_Y"],
            how="left",
            sort=False,
        )

        out = merged[[zx, zy]]

    # A left merge should produce a slope pair for every reference sensor.
    # Missing values indicate a failed identifier or coordinate alignment.
    if out.isna().any().any():
        bad = int(out.isna().any(axis=1).sum())

        raise ValueError(
            "Alignment failed: missing slopes for "
            f"{bad} sensors (NaN values after merge)."
        )

    arr = out.to_numpy(dtype=dtype)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Expected an array with shape (P_sensor, 2), but got {arr.shape}."
        )

    return arr
