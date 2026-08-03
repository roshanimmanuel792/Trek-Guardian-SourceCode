#!/usr/bin/env python3
"""
Sliding-window feature engineering for early hypoxia warning.

Rich features are used for the cloud GBM. A compact subset is used for the
on-device Decision Tree (must match firmware EWMA state).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features the ESP8266 can compute with EWMA state
FIRMWARE_FEATURES = [
    "altitude",
    "spo2",
    "heartRate",
    "spo2_ewma",
    "hr_ewma",
    "spo2_slope",
    "hr_slope",
]

# Full feature set for offline / cloud model
FULL_FEATURES = FIRMWARE_FEATURES + [
    "spo2_min",
    "spo2_std",
    "hr_std",
    "hr_spo2_ratio",
    "ascent_rate",
    "respiratory_rate",
]


def _lin_slope(y: np.ndarray) -> float:
    if len(y) < 2 or np.all(np.isnan(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    coef = np.polyfit(x[mask], y[mask], 1)
    return float(coef[0])


def build_window_features(
    df: pd.DataFrame,
    window_sec: int = 45,
    step_sec: int = 5,
    ewma_alpha: float = 0.15,
) -> pd.DataFrame:
    """
    Build features at time t from the past `window_sec` seconds.

    Assumes df is 1 Hz for a single subject, sorted by timestamp.
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    n = len(df)
    if n < window_sec + 5:
        return pd.DataFrame()

    # EWMA series aligned to each sample (causal)
    spo2_ewma = df["spo2"].ewm(alpha=ewma_alpha, adjust=False).mean()
    hr_ewma = df["heartRate"].ewm(alpha=ewma_alpha, adjust=False).mean()

    rows = []
    # End index of window is i (inclusive); window is [i-window_sec+1, i]
    for i in range(window_sec - 1, n, step_sec):
        start = i - window_sec + 1
        w = df.iloc[start : i + 1]
        spo2_w = w["spo2"].to_numpy(dtype=float)
        hr_w = w["heartRate"].to_numpy(dtype=float)
        alt_w = w["altitude"].to_numpy(dtype=float)

        spo2_now = float(df.loc[i, "spo2"])
        hr_now = float(df.loc[i, "heartRate"])
        alt_now = float(df.loc[i, "altitude"])
        spo2_e = float(spo2_ewma.iloc[i])
        hr_e = float(hr_ewma.iloc[i])

        rr = w["respiratory_rate"].mean() if "respiratory_rate" in w.columns else np.nan

        rows.append(
            {
                "subject_id": df.loc[i, "subject_id"],
                "source": df.loc[i, "source"] if "source" in df.columns else "",
                "t_index": int(i),
                "timestamp": float(df.loc[i, "timestamp"]),
                "altitude": alt_now,
                "spo2": spo2_now,
                "heartRate": hr_now,
                "spo2_ewma": spo2_e,
                "hr_ewma": hr_e,
                "spo2_slope": _lin_slope(spo2_w),
                "hr_slope": _lin_slope(hr_w),
                "spo2_min": float(np.nanmin(spo2_w)),
                "spo2_std": float(np.nanstd(spo2_w)),
                "hr_std": float(np.nanstd(hr_w)),
                "hr_spo2_ratio": hr_now / max(spo2_now, 1.0),
                "ascent_rate": _lin_slope(alt_w),
                "respiratory_rate": float(rr) if pd.notna(rr) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_features_for_all_subjects(
    vitals: pd.DataFrame,
    window_sec: int = 45,
    step_sec: int = 5,
) -> pd.DataFrame:
    parts = []
    for subject_id, sub in vitals.groupby("subject_id"):
        feats = build_window_features(sub, window_sec=window_sec, step_sec=step_sec)
        if not feats.empty:
            parts.append(feats)
            print(f"[FEATURES] {subject_id}: {len(feats)} windows")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
