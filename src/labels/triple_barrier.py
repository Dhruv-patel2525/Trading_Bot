from __future__ import annotations

import numpy as np
import pandas as pd


def _label_one_side(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    rt_cost_bps: float,
    side: str,
    tie_policy: str,
) -> np.ndarray:
    """
    Returns y in {-1,0,+1} for ONE side:
      +1 = TP first
      -1 = SL first
       0 = timeout / tie flat
    side: "long" or "short"
    tie_policy: "flat" or "worst"
    """
    n = len(open_)
    y = np.zeros(n, dtype=np.int8)

    for t in range(n):
        entry_i = t + 1
        end_i = entry_i + horizon
        if entry_i >= n or end_i > n:
            y[t] = 0
            continue
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            y[t] = 0
            continue

        entry = open_[entry_i]
        cost = entry * (rt_cost_bps / 10000.0)

        if side == "long":
            tp = entry + tp_mult * atr[t] + cost
            sl = entry - sl_mult * atr[t] - cost
            for i in range(entry_i, end_i):
                hit_tp = high[i] >= tp
                hit_sl = low[i] <= sl
                if hit_tp and hit_sl:
                    y[t] = 0 if tie_policy == "flat" else -1
                    break
                if hit_sl:
                    y[t] = -1
                    break
                if hit_tp:
                    y[t] = 1
                    break

        elif side == "short":
            tp = entry - tp_mult * atr[t] - cost
            sl = entry + sl_mult * atr[t] + cost
            for i in range(entry_i, end_i):
                hit_tp = low[i] <= tp
                hit_sl = high[i] >= sl
                if hit_tp and hit_sl:
                    y[t] = 0 if tie_policy == "flat" else -1
                    break
                if hit_sl:
                    y[t] = -1
                    break
                if hit_tp:
                    y[t] = 1
                    break
        else:
            raise ValueError("side must be 'long' or 'short'")

    return y


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon_bars: int = 24,
    tp_mult: float = 1.2,
    sl_mult: float = 1.0,
    rt_cost_bps: float = 30.0,
    tie_policy: str = "flat",
) -> pd.DataFrame:
    """
    Adds:
      - y_long: {-1,0,+1}
      - y_short: {-1,0,+1}
      - y: 3-class direction label: 1=long, -1=short, 0=no-trade (conservative)
    """
    if "atr" not in df.columns:
        raise ValueError("ATR missing. Run feature builder first.")
    if tie_policy not in ("flat", "worst"):
        raise ValueError("tie_policy must be 'flat' or 'worst'")

    out = df.copy()

    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    atr = out["atr"].to_numpy(dtype=np.float64)

    y_long = _label_one_side(
        o, h, l, atr, horizon_bars, tp_mult, sl_mult, rt_cost_bps, "long", tie_policy
    )
    y_short = _label_one_side(
        o, h, l, atr, horizon_bars, tp_mult, sl_mult, rt_cost_bps, "short", tie_policy
    )

    out["y_long"] = y_long
    out["y_short"] = y_short

    # Conservative 3-class label:
    # - take direction only if that direction hit TP (+1) AND the other direction did NOT also hit TP.
    # - otherwise label 0 (no-trade).
    y3 = np.zeros(len(out), dtype=np.int8)
    for i in range(len(out)):
        if y_long[i] == 1 and y_short[i] != 1:
            y3[i] = 1
        elif y_short[i] == 1 and y_long[i] != 1:
            y3[i] = -1
        else:
            y3[i] = 0

    out["y"] = y3
    return out
