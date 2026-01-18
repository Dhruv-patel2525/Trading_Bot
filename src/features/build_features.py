from __future__ import annotations

import numpy as np
import pandas as pd


_FEATURE_COLS = [
    "r1",
    "r2",
    "r4",
    "r8",
    "r16",
    "atr",
    "atr_bps",
    "vol32",
    "vol96",
    "volz",
    "ema20_dist",
    "ema50_dist",
    "ema20_slope",
    "ema50_slope",
    "tod_sin",
    "tod_cos",
]


def feature_cols() -> list[str]:
    return list(_FEATURE_COLS)


def _atr_segment(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def _features_one_segment(seg: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    seg = seg.copy()

    c = seg["close"].astype("float64")
    v = seg["volume"].astype("float64")

    logc = np.log(c)
    r1 = logc.diff()
    seg["r1"] = r1
    for k in [2, 4, 8, 16]:
        seg[f"r{k}"] = logc.diff(k)

    seg["atr"] = _atr_segment(seg, atr_period)
    seg["atr_bps"] = 10000.0 * (seg["atr"] / c)

    seg["vol32"] = seg["r1"].rolling(32, min_periods=32).std()
    seg["vol96"] = seg["r1"].rolling(96, min_periods=96).std()

    v_mean = v.rolling(96, min_periods=96).mean()
    v_std = v.rolling(96, min_periods=96).std()
    seg["volz"] = (v - v_mean) / v_std

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    seg["ema20_dist"] = (c / ema20) - 1.0
    seg["ema50_dist"] = (c / ema50) - 1.0
    seg["ema20_slope"] = (ema20 / ema20.shift(4)) - 1.0
    seg["ema50_slope"] = (ema50 / ema50.shift(4)) - 1.0

    ts = pd.to_datetime(seg["timestamp_ms"], unit="ms", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    ang = 2.0 * np.pi * (hour / 24.0)
    seg["tod_sin"] = np.sin(ang)
    seg["tod_cos"] = np.cos(ang)

    return seg


def build_features_segmented(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Expects df sorted by timestamp_ms and includes:
      - timestamp_ms, open, high, low, close
      - volume (already chosen by caller)
      - segment_id (continuous run id)
    """
    if "segment_id" not in df.columns:
        raise ValueError("df must contain segment_id (see dataset builder)")

    out_parts = []
    for _, seg in df.groupby("segment_id", sort=False):
        out_parts.append(_features_one_segment(seg, atr_period))

    out = (
        pd.concat(out_parts, axis=0).sort_values("timestamp_ms").reset_index(drop=True)
    )
    return out
