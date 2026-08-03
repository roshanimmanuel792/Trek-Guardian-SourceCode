#!/usr/bin/env python3
"""
Future-window hypoxia labeling (early warning task).

Labels at time t are derived from SpO2 in (t, t+horizon], NOT from current SpO2 alone.
This makes the learning problem predictive rather than circular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_LABELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def severity_from_future_spo2(future_min_spo2: float, future_mean_spo2: float) -> str:
    """
    Map future SpO2 statistics to risk tiers.

    Uses future minima (worst impending desaturation) with mean as tie-breaker context.
    """
    # Primary: worst SpO2 in the future horizon
    if future_min_spo2 < 88:
        return "CRITICAL"
    if future_min_spo2 < 90:
        return "HIGH"
    if future_min_spo2 < 93 or future_mean_spo2 < 94:
        return "MODERATE"
    return "LOW"


def attach_future_labels(
    features: pd.DataFrame,
    vitals: pd.DataFrame,
    horizon_sec: int = 90,
) -> pd.DataFrame:
    """
    For each feature row at t_index, look ahead `horizon_sec` samples (1 Hz)
    within the same subject and assign a risk label from future SpO2.
    """
    vitals = vitals.sort_values(["subject_id", "timestamp"]).copy()
    spo2_by_subject = {
        sid: g["spo2"].to_numpy(dtype=float) for sid, g in vitals.groupby("subject_id")
    }

    labels = []
    future_mins = []
    future_means = []
    keep_mask = []

    for _, row in features.iterrows():
        sid = row["subject_id"]
        i = int(row["t_index"])
        series = spo2_by_subject[sid]
        start = i + 1
        end = i + 1 + horizon_sec
        if end > len(series):
            keep_mask.append(False)
            labels.append(None)
            future_mins.append(np.nan)
            future_means.append(np.nan)
            continue

        future = series[start:end]
        if len(future) == 0 or np.all(np.isnan(future)):
            keep_mask.append(False)
            labels.append(None)
            future_mins.append(np.nan)
            future_means.append(np.nan)
            continue

        fmin = float(np.nanmin(future))
        fmean = float(np.nanmean(future))
        labels.append(severity_from_future_spo2(fmin, fmean))
        future_mins.append(fmin)
        future_means.append(fmean)
        keep_mask.append(True)

    out = features.copy()
    out["risk"] = labels
    out["future_spo2_min"] = future_mins
    out["future_spo2_mean"] = future_means
    out = out.loc[keep_mask].reset_index(drop=True)
    return out


def naive_baseline_predict(spo2_current: np.ndarray) -> np.ndarray:
    """
    Naive baseline: classify using *current* SpO2 only (no trends / altitude / HR).
    This is what a simple threshold alarm would do — ML must beat it.
    """
    preds = np.empty(len(spo2_current), dtype=object)
    for i, s in enumerate(spo2_current):
        if s < 88:
            preds[i] = "CRITICAL"
        elif s < 90:
            preds[i] = "HIGH"
        elif s < 93:
            preds[i] = "MODERATE"
        else:
            preds[i] = "LOW"
    return preds
