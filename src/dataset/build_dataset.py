from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.data.store import ParquetStore, ParquetStoreConfig
from src.utils.time import granularity_ms
from src.utils.io import write_parquet
from src.features.build_features import build_features_segmented, feature_cols
from src.labels.triple_barrier import triple_barrier_labels


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class DatasetConfig:
    exchange: str
    product_type: str
    timeframe: str
    raw_root: Path
    features_root: Path
    runs_root: Path

    atr_period: int = 14

    horizon_bars: int = 24
    tp_mult: float = 1.2
    sl_mult: float = 1.0
    rt_cost_bps: float = 30.0
    tie_policy: str = "flat"


def _canonicalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert your raw schema into a canonical schema used by features/labels.
    Uses volume_quote as 'volume' for futures.
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # Ensure sorted + deduped
    df = (
        df.sort_values("timestamp_ms")
        .drop_duplicates("timestamp_ms", keep="last")
        .reset_index(drop=True)
    )

    # Choose volume (quote) as a single column
    if "volume_quote" in df.columns:
        df["volume"] = df["volume_quote"].astype("float64")
    elif "volume" in df.columns:
        df["volume"] = df["volume"].astype("float64")
    else:
        raise ValueError("No volume_quote or volume column found")

    # Keep required cols (+ metadata)
    keep = [
        "timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "granularity",
        "source",
    ]
    for col in keep:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[keep].copy()

    # Ensure numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype("float64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")

    return df


def _add_segments_and_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Adds:
      - gap_flag: True where interval breaks
      - segment_id: increments at each gap, so rolling features don't cross gaps
    """
    df = df.copy()
    interval = granularity_ms(timeframe)
    dt = df["timestamp_ms"].diff()
    df["gap_flag"] = (dt.notna()) & (dt != interval)
    df["segment_id"] = df["gap_flag"].cumsum().astype("int64")
    return df


def _features_cache_path(cfg: DatasetConfig, symbol: str) -> Path:
    return (
        cfg.features_root
        / cfg.exchange
        / cfg.product_type
        / symbol
        / cfg.timeframe
        / "features.parquet"
    )


def load_or_build_features(
    cfg: DatasetConfig,
    store: ParquetStore,
    symbol: str,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    feat_path = _features_cache_path(cfg, symbol)
    if feat_path.exists() and not force_rebuild:
        return pd.read_parquet(feat_path)

    raw = store.read_dataset(symbol=symbol, granularity=cfg.timeframe)
    df = _canonicalize(raw)
    df = _add_segments_and_gaps(df, cfg.timeframe)

    feats = build_features_segmented(df, atr_period=cfg.atr_period)

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(feat_path, index=False)
    return feats


def build_labels_and_dataset_for_run(
    cfg: DatasetConfig,
    features_df: pd.DataFrame,
    run_dir: Path,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (labels_df, dataset_df)
    """
    labeled = triple_barrier_labels(
        features_df,
        horizon_bars=cfg.horizon_bars,
        tp_mult=cfg.tp_mult,
        sl_mult=cfg.sl_mult,
        rt_cost_bps=cfg.rt_cost_bps,
        tie_policy=cfg.tie_policy,
    )

    # Drop unusable rows:
    # - feature warmup NaNs
    # - final horizon+1 rows (cannot label reliably due to entry t+1 and horizon)
    # - rows impacted by gaps (conservative: drop any t whose future window crosses a gap)
    fcols = feature_cols()
    labeled = labeled.dropna(subset=fcols).reset_index(drop=True)

    # drop tail that can't be labeled
    cut = cfg.horizon_bars + 1
    if len(labeled) > cut:
        labeled = labeled.iloc[:-cut].reset_index(drop=True)

    # conservative gap impact removal: if a gap occurs at index i, drop t in [i-horizon, i-1]
    gap_idx = np.where(labeled["gap_flag"].to_numpy())[0]
    bad = np.zeros(len(labeled), dtype=bool)
    H = cfg.horizon_bars
    for i in gap_idx:
        start = max(0, i - H)
        end = i  # exclude i itself (new segment start) from "future crosses gap" logic
        bad[start:end] = True
        # also drop the bar right before gap from being used as "entry t+1"
        if i - 1 >= 0:
            bad[i - 1] = True

    labeled = labeled.loc[~bad].reset_index(drop=True)

    # Labels artifact
    labels_cols = ["timestamp_ms", "symbol", "y", "y_long", "y_short"]
    labels_df = labeled[labels_cols].copy()

    # Final dataset artifact: include OHLCV + features + y
    dataset_cols = (
        ["timestamp_ms", "symbol", "open", "high", "low", "close", "volume"]
        + fcols
        + ["y"]
    )
    dataset_df = labeled[dataset_cols].copy()

    labels_path = run_dir / f"labels_{symbol}_{cfg.timeframe}.parquet"
    dataset_path = run_dir / f"dataset_{symbol}_{cfg.timeframe}.parquet"
    write_parquet(labels_df, labels_path)
    write_parquet(dataset_df, dataset_path)

    return labels_df, dataset_df
