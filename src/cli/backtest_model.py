# src/cli/backtest_model.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import load

from src.features.build_features import feature_cols


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_run_dir(cfg: dict) -> Path:
    """
    Allows:
      - run_dir: "runs/...."
      - run_dir: "latest" (or omitted) -> reads runs/latest_dataset_run.txt
    """
    rd = cfg.get("run_dir")
    if rd is None or str(rd).lower() == "latest":
        runs_root = Path(cfg.get("runs_root", "runs"))
        pointer = runs_root / "latest_dataset_run.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"run_dir not provided (or 'latest') but pointer missing: {pointer}. "
                "Run build-dataset first."
            )
        rd = pointer.read_text(encoding="utf-8").strip()
        if not rd:
            raise ValueError(f"Pointer file is empty: {pointer}")

    run_dir = Path(str(rd))
    if not run_dir.exists():
        raise FileNotFoundError(f"Resolved run_dir does not exist: {run_dir}")
    return run_dir


def _proba_to_label_and_conf(proba_row: np.ndarray) -> Tuple[int, float]:
    """
    Training mapping:
      class 0 -> -1 (short)
      class 1 ->  0 (no-trade)
      class 2 -> +1 (long)
    Returns (label in {-1,0,+1}, confidence=max prob).
    """
    cls = int(np.argmax(proba_row))
    conf = float(proba_row[cls])
    if cls == 0:
        return -1, conf
    if cls == 1:
        return 0, conf
    return 1, conf


def _position_size_qty(
    equity: float,
    entry: float,
    sl_price: float,
    risk_pct: float,
    max_leverage: float,
) -> float:
    """
    Risk-based sizing with leverage cap.
      risk_usd = equity * (risk_pct/100)
      stop_dist = abs(entry - sl_price)
      qty = risk_usd / stop_dist
    cap notional <= equity * max_leverage
    """
    risk_usd = equity * (risk_pct / 100.0)
    stop_dist = abs(entry - sl_price)
    if stop_dist <= 0:
        return 0.0

    qty = risk_usd / stop_dist

    max_notional = equity * max_leverage
    notional = qty * entry
    if notional > max_notional:
        qty = max_notional / entry

    return float(qty)


def _simulate_exit(
    side: int,
    t: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    rt_cost_bps: float,
    tie_policy: str,
) -> Tuple[int, float, str]:
    """
    Simulate SL/TP intrabar.

    Signal at index t -> entry at t+1 open.

    Cost-aware barriers (same scheme as labeler):
      cost = entry * rt_cost_bps / 10000

    LONG:
      tp = entry + tp_mult*atr[t] + cost
      sl = entry - sl_mult*atr[t] - cost

    SHORT:
      tp = entry - tp_mult*atr[t] - cost
      sl = entry + sl_mult*atr[t] + cost

    tie_policy:
      - "worst": if TP and SL hit in same candle, exit at SL (worst-case)
      - "flat":  if TP and SL hit in same candle, treat as TIMEOUT (exit at horizon end)
    """
    n = len(open_)
    entry_i = t + 1
    if entry_i >= n:
        return n - 1, float(open_[n - 1]), "NOENTRY"

    entry = float(open_[entry_i])
    cost = entry * (rt_cost_bps / 10000.0)

    if not np.isfinite(atr[t]) or atr[t] <= 0:
        end_i = min(n - 1, entry_i + horizon - 1)
        return end_i, float(open_[end_i]), "TIME_ATR0"

    end = min(n, entry_i + horizon)

    if side == 1:  # LONG
        tp = entry + tp_mult * float(atr[t]) + cost
        sl = entry - sl_mult * float(atr[t]) - cost

        for i in range(entry_i, end):
            hit_tp = float(high[i]) >= tp
            hit_sl = float(low[i]) <= sl

            if hit_tp and hit_sl:
                if tie_policy == "worst":
                    return i, sl, "SL_TIE"
                else:
                    end_i = end - 1
                    return end_i, float(open_[end_i]), "TIME_TIE"

            if hit_sl:
                return i, sl, "SL"
            if hit_tp:
                return i, tp, "TP"

        end_i = end - 1
        return end_i, float(open_[end_i]), "TIME"

    else:  # SHORT
        tp = entry - tp_mult * float(atr[t]) - cost
        sl = entry + sl_mult * float(atr[t]) + cost

        for i in range(entry_i, end):
            hit_tp = float(low[i]) <= tp
            hit_sl = float(high[i]) >= sl

            if hit_tp and hit_sl:
                if tie_policy == "worst":
                    return i, sl, "SL_TIE"
                else:
                    end_i = end - 1
                    return end_i, float(open_[end_i]), "TIME_TIE"

            if hit_sl:
                return i, sl, "SL"
            if hit_tp:
                return i, tp, "TP"

        end_i = end - 1
        return end_i, float(open_[end_i]), "TIME"


def _load_inputs(
    run_dir: Path, symbol: str, timeframe: str
) -> Tuple[pd.DataFrame, Any]:
    ds_path = run_dir / f"dataset_{symbol}_{timeframe}.parquet"
    model_path = run_dir / "models" / f"model_{symbol}_{timeframe}.joblib"

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset missing: {ds_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model missing: {model_path}. "
            f"Make sure training ran on the same run_dir."
        )

    df = pd.read_parquet(ds_path).sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    model = load(model_path)
    return df, model


def backtest_symbol(
    cfg: dict, run_dir: Path, symbol: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tf = str(cfg["timeframe"])
    df, model = _load_inputs(run_dir, symbol, tf)

    # features
    cols = feature_cols()
    df = df.dropna(subset=cols + ["open", "high", "low", "atr"]).copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    X = df[cols].to_numpy(dtype=np.float64)
    proba = model.predict_proba(X)

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr = df["atr"].to_numpy(dtype=np.float64)

    pth = float(cfg["prob_threshold"])

    ex = cfg["execution"]
    horizon = int(ex["horizon_bars"])
    tp_mult = float(ex["tp_mult"])
    sl_mult = float(ex["sl_mult"])
    rt_cost_bps = float(ex["rt_cost_bps"])
    tie_policy = str(ex.get("tie_policy", "worst")).lower()

    rk = cfg["risk"]
    equity = float(rk["starting_equity"])
    start_equity = equity
    risk_pct = float(rk["risk_per_trade_pct"])
    daily_max_loss_pct = float(rk["daily_max_loss_pct"])
    max_leverage = float(rk.get("max_leverage", 10))
    one_position = bool(rk.get("one_position", True))

    trades: List[Dict[str, Any]] = []

    n = len(df)
    i = 0

    # day tracking for daily stop
    cur_day = None
    day_start_equity = equity
    day_pnl = 0.0

    while i < n - (horizon + 2):
        ts = df.loc[i, "timestamp"]
        d = ts.date()

        if cur_day is None or d != cur_day:
            cur_day = d
            day_start_equity = equity
            day_pnl = 0.0

        # daily kill switch
        daily_limit = day_start_equity * (daily_max_loss_pct / 100.0)
        if day_pnl <= -daily_limit:
            i += 1
            continue

        pred, conf = _proba_to_label_and_conf(proba[i])

        # gate
        if pred == 0 or conf < pth:
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= n:
            break

        entry_price = float(open_[entry_i])

        # stop price for sizing, consistent with barrier definition
        cost = entry_price * (rt_cost_bps / 10000.0)
        a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
        if a <= 0:
            i += 1
            continue

        if pred == 1:  # long
            sl_price = entry_price - sl_mult * a - cost
        else:  # short
            sl_price = entry_price + sl_mult * a + cost

        qty = _position_size_qty(equity, entry_price, sl_price, risk_pct, max_leverage)
        if qty <= 0:
            i += 1
            continue

        exit_i, exit_price, exit_reason = _simulate_exit(
            side=pred,
            t=i,
            open_=open_,
            high=high,
            low=low,
            atr=atr,
            horizon=horizon,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            rt_cost_bps=rt_cost_bps,
            tie_policy=tie_policy,
        )

        # PnL: costs are already embedded in barriers; don't subtract again.
        if pred == 1:
            pnl = qty * (float(exit_price) - entry_price)
        else:
            pnl = qty * (entry_price - float(exit_price))

        equity += pnl
        day_pnl += pnl

        trades.append(
            {
                "symbol": symbol,
                "side": "LONG" if pred == 1 else "SHORT",
                "signal_ts": df.loc[i, "timestamp"],
                "entry_ts": df.loc[entry_i, "timestamp"],
                "exit_ts": df.loc[int(exit_i), "timestamp"],
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "qty": float(qty),
                "pnl_usd": float(pnl),
                "equity_after": float(equity),
                "prob": float(conf),
                "exit_reason": exit_reason,
            }
        )

        # enforce one-position by jumping to exit
        if one_position:
            i = max(i + 1, int(exit_i))
        else:
            i += 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        summary = {
            "symbol": symbol,
            "trades": 0,
            "net_pnl_usd": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_pnl_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "prob_threshold": pth,
            "resolved_run_dir": str(run_dir),
        }
        return trades_df, summary

    eq = trades_df["equity_after"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd_pct = float(dd.min()) * 100.0

    pnl = trades_df["pnl_usd"].to_numpy(dtype=np.float64)
    win_rate_pct = float((pnl > 0).mean()) * 100.0

    net_pnl = float(equity - start_equity)
    ret_pct = (net_pnl / start_equity) * 100.0

    summary = {
        "symbol": symbol,
        "trades": int(len(trades_df)),
        "net_pnl_usd": net_pnl,
        "return_pct": ret_pct,
        "win_rate_pct": win_rate_pct,
        "avg_pnl_usd": float(pnl.mean()),
        "max_drawdown_pct": max_dd_pct,
        "prob_threshold": pth,
        "execution": cfg["execution"],
        "risk": cfg["risk"],
        "resolved_run_dir": str(run_dir),
    }
    return trades_df, summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/backtest_model.yaml")
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    run_dir = resolve_run_dir(cfg)

    out_subdir = str(cfg.get("output", {}).get("subdir", "backtests"))
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for sym in cfg["symbols"]:
        trades_df, summary = backtest_symbol(cfg, run_dir, sym)

        trades_path = out_dir / f"trades_{sym}_{cfg['timeframe']}.parquet"
        trades_df.to_parquet(trades_path, index=False)

        summary_path = out_dir / f"summary_{sym}_{cfg['timeframe']}.yaml"
        summary_path.write_text(
            yaml.safe_dump(summary, sort_keys=False),
            encoding="utf-8",
        )

        print(
            f"[{sym}] trades={summary['trades']} "
            f"net_pnl={summary['net_pnl_usd']:.2f} "
            f"ret={summary['return_pct']:.2f}% "
            f"win={summary['win_rate_pct']:.1f}% "
            f"maxDD={summary['max_drawdown_pct']:.2f}%"
        )
        print(f"[{sym}] wrote: {trades_path}")

        summaries.append(summary)

    all_path = out_dir / f"summary_ALL_{cfg['timeframe']}.yaml"
    all_path.write_text(
        yaml.safe_dump({"summaries": summaries}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[done] wrote: {all_path}")


if __name__ == "__main__":
    main()
