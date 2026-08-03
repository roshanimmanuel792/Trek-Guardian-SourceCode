#!/usr/bin/env python3
"""
Preprocess Harespod + HAPP into a unified 1 Hz time-series table.

Output columns:
  subject_id, timestamp, altitude, spo2, heartRate, respiratory_rate, source
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "dataset" / "raw"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"

# Harespod Data_Disc altitude codes (filename suffix) -> meters
HARESPOD_ALTITUDE_MAP = {
    "20": 2000.0,
    "25": 2500.0,
    "30": 3000.0,
    "35": 3500.0,
    "40": 4000.0,
}


def _clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    df["heartRate"] = pd.to_numeric(df["heartRate"], errors="coerce")
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")

    if "respiratory_rate" in df.columns:
        df["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")
    else:
        df["respiratory_rate"] = np.nan

    df.loc[(df["spo2"] < 50) | (df["spo2"] > 100), "spo2"] = np.nan
    df.loc[(df["heartRate"] < 30) | (df["heartRate"] > 220), "heartRate"] = np.nan
    df.loc[(df["altitude"] < 0) | (df["altitude"] > 9000), "altitude"] = np.nan

    df = df.dropna(subset=["spo2", "heartRate", "altitude"])
    return df


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    """Parse epoch numbers, relative seconds, or clock strings (HH:MM:SS)."""
    if pd.api.types.is_numeric_dtype(raw):
        ts = raw.astype(float)
        if ts.max() > 1e12:
            return pd.to_datetime(ts, unit="ms", errors="coerce")
        if ts.max() > 1e10:
            return pd.to_datetime(ts, unit="ms", errors="coerce")
        if ts.max() > 1e9:
            return pd.to_datetime(ts, unit="s", errors="coerce")
        return pd.to_datetime(ts - ts.min(), unit="s", errors="coerce")

    as_str = raw.astype(str)
    dt = pd.to_datetime(as_str, format="%H:%M:%S", errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(as_str, errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime("2000-01-01 " + as_str, errors="coerce")
    else:
        # Pure clock times need a date anchor for resampling
        if dt.dt.year.min() < 1971:
            dt = pd.to_datetime("2000-01-01 " + as_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            if dt.notna().sum() == 0:
                dt = pd.to_datetime("2000-01-01 " + as_str, errors="coerce")

    if dt.notna().any():
        delta = dt.diff().dt.total_seconds()
        wraps = delta < -12 * 3600
        if wraps.any():
            offsets = wraps.cumsum()
            dt = dt + pd.to_timedelta(offsets, unit="D")
    return dt


def _resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Resample irregular / high-rate series to 1 Hz mean values."""
    df = df.copy()
    if df.empty:
        return df

    dt = _parse_timestamps(df["timestamp"])
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt").set_index("_dt")
    numeric_cols = ["altitude", "spo2", "heartRate", "respiratory_rate"]
    present = [c for c in numeric_cols if c in df.columns]
    resampled = df[present].resample("1s").mean()
    resampled = resampled.interpolate(limit=5).dropna(subset=["spo2", "heartRate"])
    resampled = resampled.reset_index().rename(columns={"_dt": "timestamp"})
    resampled["timestamp"] = (
        resampled["timestamp"] - resampled["timestamp"].iloc[0]
    ).dt.total_seconds()
    return resampled


def load_happ_subject(csv_path: Path) -> pd.DataFrame | None:
    """Load one HAPP subject CSV into the unified schema."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {csv_path.name}: {exc}")
        return None

    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    spo2_col = pick("spo2", "spo₂", "sp02", "oxygen_saturation", "spo2(%)")
    hr_col = pick("heartrate", "heart_rate", "heart rate", "hr", "pulse", "pulse_rate")
    alt_col = pick("altitude", "alt", "height", "baro_altitude")
    rr_col = pick(
        "respiratory_rate",
        "respiratory rate",
        "rr",
        "respiration",
        "resp_rate",
        "breathing_rate",
    )
    ts_col = pick("timestamp", "time", "t", "datetime")

    if spo2_col is None:
        for c in df.columns:
            if "spo" in c.lower():
                spo2_col = c
                break
    if hr_col is None:
        for c in df.columns:
            cl = c.lower()
            if "heart" in cl or cl == "hr" or "pulse" in cl:
                hr_col = c
                break
    if alt_col is None:
        for c in df.columns:
            if "alt" in c.lower():
                alt_col = c
                break
    if rr_col is None:
        for c in df.columns:
            cl = c.lower()
            if "resp" in cl or "breath" in cl:
                rr_col = c
                break
    if ts_col is None:
        for c in df.columns:
            if "time" in c.lower():
                ts_col = c
                break

    if spo2_col is None or hr_col is None or alt_col is None:
        print(f"[WARN] Missing required columns in {csv_path.name}: {list(df.columns)}")
        return None

    out = pd.DataFrame(
        {
            "timestamp": df[ts_col] if ts_col else np.arange(len(df), dtype=float),
            "altitude": df[alt_col],
            "spo2": df[spo2_col],
            "heartRate": df[hr_col],
            "respiratory_rate": df[rr_col] if rr_col else np.nan,
        }
    )
    out = _clean_vitals(out)
    out = _resample_1hz(out)
    if out.empty:
        return None

    subject_id = f"happ_{csv_path.stem}"
    out["subject_id"] = subject_id
    out["source"] = "happ"
    return out[
        ["subject_id", "timestamp", "altitude", "spo2", "heartRate", "respiratory_rate", "source"]
    ]


def _find_harespod_subject_dirs(disc_root: Path) -> list[Path]:
    """Locate per-subject folders under extracted Data_Disc."""
    if not disc_root.exists():
        return []

    candidates = []
    for path in disc_root.rglob("*"):
        if not path.is_dir():
            continue
        if any(path.glob("spv_*.csv")):
            candidates.append(path)

    return sorted(set(candidates))


def load_harespod_subject(subject_dir: Path) -> pd.DataFrame | None:
    """
    Load Harespod Data_Disc subject.

    Published Data_Disc values are normalized (Scientific Data 2024).
    Invert with fixed scales from the paper: SpO2∈[0,100], HR∈[0,250].
    """
    frames = []
    for code, altitude_m in HARESPOD_ALTITUDE_MAP.items():
        spo2_path = subject_dir / f"spv_{code}.csv"
        hr_path = subject_dir / f"hr_{code}.csv"
        prt_path = subject_dir / f"prt_{code}.csv"

        if not spo2_path.exists():
            continue

        spo2 = pd.read_csv(spo2_path, header=None, names=["timestamp", "value"])
        seg = pd.DataFrame(
            {
                "timestamp": spo2["timestamp"],
                "spo2": pd.to_numeric(spo2["value"], errors="coerce"),
                "altitude": altitude_m,
            }
        )

        hr_file = hr_path if hr_path.exists() else prt_path
        if hr_file.exists():
            hr = pd.read_csv(hr_file, header=None, names=["timestamp", "value"])
            hr = pd.DataFrame(
                {
                    "timestamp": hr["timestamp"].astype(str),
                    "heartRate": pd.to_numeric(hr["value"], errors="coerce"),
                }
            )
            seg["timestamp"] = seg["timestamp"].astype(str)
            seg = seg.merge(hr, on="timestamp", how="left")
        else:
            seg["heartRate"] = np.nan

        seg["respiratory_rate"] = np.nan
        frames.append(seg)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    if df["spo2"].max(skipna=True) <= 1.5:
        df["spo2"] = df["spo2"] * 100.0
    if df["heartRate"].max(skipna=True) <= 2.0:
        df["heartRate"] = df["heartRate"] * 250.0

    df = _clean_vitals(df)
    blocks = []
    for alt, block in df.groupby("altitude", sort=True):
        block = block.copy()
        block = _resample_1hz(block)
        if block.empty:
            continue
        block["altitude"] = alt
        blocks.append(block)

    if not blocks:
        return None

    t = 0.0
    stamped = []
    for alt, block in pd.concat(blocks, ignore_index=True).groupby("altitude", sort=True):
        b = block.copy().reset_index(drop=True)
        b["timestamp"] = np.arange(len(b), dtype=float) + t
        t = float(b["timestamp"].iloc[-1]) + 1.0
        stamped.append(b)
    out = pd.concat(stamped, ignore_index=True)

    subject_id = f"harespod_{subject_dir.name}"
    out["subject_id"] = subject_id
    out["source"] = "harespod"
    return out[
        ["subject_id", "timestamp", "altitude", "spo2", "heartRate", "respiratory_rate", "source"]
    ]


def preprocess_all() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    happ_dir = RAW_DIR / "happ"
    if happ_dir.exists():
        for csv_path in sorted(happ_dir.glob("*.csv")):
            print(f"[HAPP] {csv_path.name}")
            subj = load_happ_subject(csv_path)
            if subj is not None and len(subj) > 30:
                frames.append(subj)
                print(f"       -> {len(subj)} samples")
            else:
                print("       -> skipped (insufficient clean samples)")
    else:
        print(f"[WARN] Missing {happ_dir} — run download_datasets.py")

    harespod_disc = RAW_DIR / "harespod" / "Data_Disc"
    search_roots = [harespod_disc]
    nested = RAW_DIR / "harespod"
    if nested.exists():
        search_roots.extend([p for p in nested.rglob("Data_Disc") if p.is_dir()])

    subject_dirs: list[Path] = []
    for root in search_roots:
        subject_dirs.extend(_find_harespod_subject_dirs(root))
    subject_dirs = sorted(set(subject_dirs))

    if subject_dirs:
        for sdir in subject_dirs:
            print(f"[HARESPOD] {sdir.name}")
            subj = load_harespod_subject(sdir)
            if subj is not None and len(subj) > 30:
                frames.append(subj)
                print(f"           -> {len(subj)} samples")
            else:
                print("           -> skipped")
    else:
        print("[WARN] No Harespod subject folders found. Run: python download_datasets.py")

    if not frames:
        raise SystemExit(
            "No real data available after preprocessing.\n"
            "Run: python download_datasets.py\n"
            "Then: python preprocess.py"
        )

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "vitals_1hz.parquet"
    csv_path = PROCESSED_DIR / "vitals_1hz.csv"
    combined.to_parquet(out_path, index=False)
    combined.to_csv(csv_path, index=False)

    summary = {
        "n_rows": int(len(combined)),
        "n_subjects": int(combined["subject_id"].nunique()),
        "sources": combined["source"].value_counts().to_dict(),
        "subjects": sorted(combined["subject_id"].unique().tolist()),
        "spo2_range": [float(combined["spo2"].min()), float(combined["spo2"].max())],
        "altitude_range": [float(combined["altitude"].min()), float(combined["altitude"].max())],
    }
    (PROCESSED_DIR / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=" * 60)
    print(f"[OK] Wrote {out_path} and {csv_path}")
    print(json.dumps(summary, indent=2))
    return out_path


def main() -> None:
    print("=" * 60)
    print("Trek Guardian — Preprocess Real Datasets")
    print("=" * 60)
    preprocess_all()


if __name__ == "__main__":
    main()
