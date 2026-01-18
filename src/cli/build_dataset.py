# src/cli/build_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from traitlets import List
import yaml

from src.data.store import ParquetStore, ParquetStoreConfig
from src.dataset.build_dataset import (
    DatasetConfig,
    load_or_build_features,
    build_labels_and_dataset_for_run,
)


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_resolved_config(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def write_latest_pointer(runs_root: Path, run_dir: Path) -> None:
    """
    Writes a pointer file so training/backtest can automatically find the latest dataset run.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "latest_dataset_run.txt").write_text(str(run_dir), encoding="utf-8")


def _tf_minutes(tf: str) -> int:
    tf = str(tf)
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith(("H", "h")):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe for horizon conversion: {tf}")


def _horizon_to_bars(horizon_cfg: dict, timeframe: str) -> int:
    unit = str(horizon_cfg.get("unit", "bars")).lower()
    value = float(horizon_cfg.get("value"))
    if unit in ("bars", "bar"):
        return int(value)
    if unit in ("hours", "hour", "h"):
        tf_min = _tf_minutes(timeframe)
        bars_per_hour = 60.0 / tf_min
        return int(round(value * bars_per_hour))
    raise ValueError(f"Unsupported horizon unit: {unit}")


def _parse_strategy_yaml(strategy: dict) -> dict:
    """
    Convert nested strategy.yaml into flat keys used by DatasetConfig.
    Also validates unsupported modes to avoid silent misconfiguration.
    """
    if "timeframe" not in strategy:
        raise ValueError("strategy.yaml must contain 'timeframe'")

    timeframe = strategy["timeframe"]

    # horizon -> bars
    horizon_cfg = strategy.get("horizon", {"unit": "hours", "value": 6})
    horizon_bars = _horizon_to_bars(horizon_cfg, timeframe)

    # costs
    rt_cost_bps = float(strategy.get("costs", {}).get("rt_cost_bps", 30.0))

    # tie handling
    tie_policy = str(strategy.get("tie_policy", "flat"))

    # barriers
    barriers = strategy.get("barriers", {})
    mode = str(barriers.get("mode", "atr")).lower()

    if mode == "atr":
        atr_cfg = barriers.get("atr", {})
        atr_period = int(atr_cfg.get("period", 14))
        tp_mult = float(atr_cfg.get("tp_mult", 1.2))
        sl_mult = float(atr_cfg.get("sl_mult", 1.0))
    elif mode == "percent":
        raise ValueError(
            "barriers.mode='percent' is not supported by the current labeler. "
            "Use barriers.mode='atr' for now or implement percent-mode labeling first."
        )
    else:
        raise ValueError(f"Unsupported barriers.mode: {mode}")

    return {
        "timeframe": timeframe,
        "horizon_bars": horizon_bars,
        "atr_period": atr_period,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "rt_cost_bps": rt_cost_bps,
        "tie_policy": tie_policy,
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True, help="Path to dataset.yaml or strategy.yaml"
    )
    ap.add_argument("--symbol", default=None, help="Optional: build only one symbol")
    ap.add_argument(
        "--force-features", action="store_true", help="Rebuild features cache"
    )
    args = ap.parse_args(argv)

    cfg_dict = load_yaml(Path(args.config))

    # If it's a nested strategy.yaml, flatten it to expected keys
    if "barriers" in cfg_dict or "horizon" in cfg_dict:
        flat = _parse_strategy_yaml(cfg_dict)

        # Provide defaults if missing (recommended: include these in strategy.yaml)
        cfg_dict.setdefault("exchange", "bitget")
        cfg_dict.setdefault("product_type", "usdt-futures")
        cfg_dict.setdefault("symbols", ["BTCUSDT"])
        cfg_dict.setdefault("raw_root", "data/raw")
        cfg_dict.setdefault("features_root", "data/processed/features")
        cfg_dict.setdefault("runs_root", "runs")

        # Merge derived flat fields into cfg_dict for transparency in config_resolved.yaml
        cfg_dict = {**cfg_dict, **flat}

    # Optional CLI override
    if args.symbol is not None:
        cfg_dict["symbols"] = [args.symbol]

    # Build DatasetConfig from the (now-flat) cfg_dict
    cfg = DatasetConfig(
        exchange=cfg_dict["exchange"],
        product_type=cfg_dict["product_type"],
        timeframe=cfg_dict["timeframe"],
        raw_root=Path(cfg_dict.get("raw_root", "data/raw")),
        features_root=Path(cfg_dict.get("features_root", "data/processed/features")),
        runs_root=Path(cfg_dict.get("runs_root", "runs")),
        atr_period=int(cfg_dict.get("atr_period", 14)),
        horizon_bars=int(cfg_dict.get("horizon_bars", 24)),
        tp_mult=float(cfg_dict.get("tp_mult", 1.2)),
        sl_mult=float(cfg_dict.get("sl_mult", 1.0)),
        rt_cost_bps=float(cfg_dict.get("rt_cost_bps", 30.0)),
        tie_policy=str(cfg_dict.get("tie_policy", "flat")),
    )

    run_dir = cfg.runs_root / now_run_id()
    save_resolved_config(run_dir, cfg_dict)

    store = ParquetStore(
        ParquetStoreConfig(
            root=cfg.raw_root,
            source=cfg.exchange,
            product_type=cfg.product_type,
        )
    )

    for sym in cfg_dict["symbols"]:
        feats = load_or_build_features(
            cfg, store, sym, force_rebuild=args.force_features
        )
        labels_df, dataset_df = build_labels_and_dataset_for_run(
            cfg, feats, run_dir, sym
        )

        vc = dataset_df["y"].value_counts(normalize=True).sort_index()
        print(f"[{sym}] dataset_rows={len(dataset_df)} label_dist={vc.to_dict()}")

    print(f"[run] artifacts written to: {run_dir}")

    # NEW: write pointer to latest dataset run
    write_latest_pointer(cfg.runs_root, run_dir)
    print(f"[run] updated pointer: {cfg.runs_root / 'latest_dataset_run.txt'}")


if __name__ == "__main__":
    main()
