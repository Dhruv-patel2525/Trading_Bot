# src/cli/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
from joblib import dump

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
                f"run_dir not provided (or 'latest') but pointer file not found: {pointer}. "
                "Run build-dataset first so it writes latest_dataset_run.txt."
            )
        rd = pointer.read_text(encoding="utf-8").strip()
        if not rd:
            raise ValueError(f"Pointer file is empty: {pointer}")
    run_dir = Path(str(rd))
    if not run_dir.exists():
        raise FileNotFoundError(f"Resolved run_dir does not exist: {run_dir}")
    return run_dir


def _load_symbol_dataset(run_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = run_dir / f"dataset_{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)

    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def _map_labels_to_012(y: np.ndarray) -> np.ndarray:
    """
    Map y in {-1,0,+1} -> {0,1,2} for sklearn multiclass.
      -1 -> 0, 0 -> 1, +1 -> 2
    """
    out = np.empty_like(y, dtype=np.int64)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def _inverse_map_proba_to_label(proba: np.ndarray) -> np.ndarray:
    """
    proba columns correspond to classes {0,1,2} -> labels {-1,0,+1}.
    """
    cls = proba.argmax(axis=1)
    return np.where(cls == 0, -1, np.where(cls == 1, 0, 1))


def _class_weights_from_y(y012: np.ndarray) -> Dict[int, float]:
    vals, counts = np.unique(y012, return_counts=True)
    total = counts.sum()
    return {int(v): float(total / (len(vals) * c)) for v, c in zip(vals, counts)}


def _time_split_indices(
    n: int, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[int, int]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    i_train_end = int(round(n * train_frac))
    i_val_end = int(round(n * (train_frac + val_frac)))
    return i_train_end, i_val_end


def _apply_embargo(
    df: pd.DataFrame, i_train_end: int, i_val_end: int, embargo_bars: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Removes embargo_bars around split boundaries to prevent leakage because labels look ahead.
    """
    n = len(df)
    tr_end = max(0, i_train_end - embargo_bars)
    va_start = min(n, i_train_end + embargo_bars)

    va_end = max(va_start, i_val_end - embargo_bars)
    te_start = min(n, i_val_end + embargo_bars)

    train = df.iloc[:tr_end].copy()
    val = df.iloc[va_start:va_end].copy()
    test = df.iloc[te_start:].copy()
    return train, val, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean())
    pred_trade = y_pred != 0
    if pred_trade.any():
        trade_acc = float((y_true[pred_trade] == y_pred[pred_trade]).mean())
        trade_rate = float(pred_trade.mean())
    else:
        trade_acc = 0.0
        trade_rate = 0.0
    return {"acc": acc, "trade_acc": trade_acc, "trade_rate": trade_rate}


def train_one_symbol(
    df: pd.DataFrame,
    target_col: str,
    embargo_bars: int,
    split_cfg: dict,
    model_cfg: dict,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = df.dropna(subset=feature_cols() + [target_col]).copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    n = len(df)
    i_train_end, i_val_end = _time_split_indices(
        n,
        float(split_cfg["train_frac"]),
        float(split_cfg["val_frac"]),
        float(split_cfg["test_frac"]),
    )
    train_df, val_df, test_df = _apply_embargo(df, i_train_end, i_val_end, embargo_bars)

    if len(train_df) < 500:
        raise ValueError(f"Train set too small after embargo: {len(train_df)} rows")

    X_train = train_df[feature_cols()].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols()].to_numpy(dtype=np.float64) if len(val_df) else None
    X_test = test_df[feature_cols()].to_numpy(dtype=np.float64)

    y_train = _map_labels_to_012(train_df[target_col].to_numpy(dtype=np.int64))
    y_val = (
        _map_labels_to_012(val_df[target_col].to_numpy(dtype=np.int64))
        if len(val_df)
        else None
    )
    y_test = _map_labels_to_012(test_df[target_col].to_numpy(dtype=np.int64))

    # class weights -> sample weights
    cw = _class_weights_from_y(y_train)
    w_train = np.array([cw[int(c)] for c in y_train], dtype=np.float64)

    model = HistGradientBoostingClassifier(
        max_iter=int(model_cfg.get("max_iter", 400)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        l2_regularization=float(model_cfg.get("l2_regularization", 1.0)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 30)),
        random_state=int(model_cfg.get("random_state", 42)),
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate on test
    p_test = model.predict_proba(X_test)
    y_pred_test = _inverse_map_proba_to_label(p_test)
    y_true_test = test_df[target_col].to_numpy(dtype=np.int64)
    test_metrics = _metrics(y_true_test, y_pred_test)

    # Evaluate on val (optional)
    val_metrics = {}
    if X_val is not None and len(val_df):
        p_val = model.predict_proba(X_val)
        y_pred_val = _inverse_map_proba_to_label(p_val)
        y_true_val = val_df[target_col].to_numpy(dtype=np.int64)
        val_metrics = _metrics(y_true_val, y_pred_val)

    info = {
        "rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": n,
        },
        "metrics": {"val": val_metrics, "test": test_metrics},
        "class_weight_train_012": cw,
        "feature_cols": feature_cols(),
        "split": split_cfg,
        "embargo_bars": embargo_bars,
    }
    return model, info


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/train.yaml")
    args = ap.parse_args(argv)

    cfg = load_yaml(Path(args.config))

    run_dir = resolve_run_dir(cfg)
    timeframe = str(cfg["timeframe"])
    symbols = list(cfg["symbols"])
    target_col = str(cfg.get("target_col", "y"))

    embargo_bars = int(cfg.get("embargo_bars", cfg.get("horizon_bars", 24)))
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]

    out_dir = run_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for sym in symbols:
        df = _load_symbol_dataset(run_dir, sym, timeframe)

        model, info = train_one_symbol(
            df=df,
            target_col=target_col,
            embargo_bars=embargo_bars,
            split_cfg=split_cfg,
            model_cfg=model_cfg,
        )

        model_path = out_dir / f"model_{sym}_{timeframe}.joblib"
        dump(model, model_path)

        meta_path = out_dir / f"model_{sym}_{timeframe}.yaml"
        meta = {
            "symbol": sym,
            "timeframe": timeframe,
            "target_col": target_col,
            "model_type": model_cfg.get("type", "sklearn_hgb"),
            "model_path": str(model_path),
            "train_info": info,
            "resolved_run_dir": str(run_dir),
        }
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

        results[sym] = info["metrics"]["test"]
        print(f"[{sym}] saved: {model_path}")
        print(f"[{sym}] test_metrics: {info['metrics']['test']}")

    print(f"[done] models saved under: {out_dir}")
    print(f"[summary] {results}")


if __name__ == "__main__":
    main()
