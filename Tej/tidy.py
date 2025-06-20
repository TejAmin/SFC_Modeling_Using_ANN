# ----------------------------------------------------------------------
#  Feature-engineering / de-correlation utility
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd


def tidy_features(
    df: pd.DataFrame,
    *,
    corr_thresh: float = 0.95,
    spread_name: str = "spread",
    skew_name: str = "skew",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Return a copy of *df* with:
      • new columns <spread_name>, <skew_name> derived from d10/d50/d90
      • original d10, d50, d90 dropped
      • any other numeric columns with |corr| > corr_thresh removed

    Parameters
    ----------
    corr_thresh : float
        Absolute Pearson correlation above which a column is considered redundant.
    eps : float
        Small constant to prevent divide-by-zero in the skew calculation.

    Notes
    -----
    The algorithm for dropping correlated features:
        * Build the absolute correlation matrix of numeric columns.
        * Walk the upper triangle; when |corr| > thresh, mark the *later* column.
        * Finally drop all marked columns.
    """
    df = df.copy()

    # ── Combine percentiles into spread & skew ─────────────────────────
    if {"d10", "d50", "d90"}.issubset(df.columns):
        d10, d50, d90 = df["d10"], df["d50"], df["d90"]
        df[spread_name] = d90 - d10
        df[skew_name] = (d50 - d10) / (d90 - d50 + eps)
        df.drop(columns=["d10", "d50", "d90"], inplace=True)
    else:
        missing = {"d10", "d50", "d90"} - set(df.columns)
        print(f"Warning: missing percentile columns: {missing} — no skew/spread created.")

    # ── Drop highly correlated features ───────────────────────────────
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]

    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Dropped {len(to_drop)} highly correlated feature(s): {to_drop}")

    return df

