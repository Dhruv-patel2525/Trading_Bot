from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import yaml

from dotenv import load_dotenv
import os

from src.data.bitget_client import BitgetClientConfig, BitgetHttpClient
from src.data.bitget_candles import CandleFetchConfig, BitgetCandleFetcher
from src.data.store import ParquetStore, ParquetStoreConfig
from src.data.validate import validate_ohlcv
from src.utils.time import utcnow_ms, granularity_ms


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_resolved_config(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    load_dotenv()  # reads .env

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--tf", default=None)
    ap.add_argument(
        "--symbol",
        default=None,
        help="Optional: fetch a single symbol (overrides symbols list)",
    )
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    # CLI overrides (highest priority)
    if args.days is not None:
        cfg["days"] = args.days
    if args.tf is not None:
        cfg["timeframe"] = args.tf
    if args.symbol is not None:
        cfg["symbols"] = [args.symbol]

    # Env overrides (good for machine-specific things)
    base_url = os.getenv("BITGET_BASE_URL", "https://api.bitget.com")

    run_dir = Path("runs") / now_run_id()
    save_resolved_config(run_dir, {**cfg, "BITGET_BASE_URL": base_url})

    client = BitgetHttpClient(BitgetClientConfig(base_url=base_url))
    fetcher = BitgetCandleFetcher(client)

    tf = cfg["timeframe"]
    interval = granularity_ms(tf)
    end_ms = utcnow_ms()
    target_bars = int((cfg["days"] * 24 * 60 * 60_000) // interval)

    store = ParquetStore(
        ParquetStoreConfig(
            root=Path(cfg.get("out_root", "data/raw")),
            source=cfg.get("exchange", "bitget"),
            product_type=cfg.get("product_type", "usdt-futures"),
        )
    )

    for sym in cfg["symbols"]:
        candle_cfg = CandleFetchConfig(
            symbol=sym,
            granularity=tf,
            product_type=cfg.get("product_type", "usdt-futures"),
            limit=int(cfg.get("limit", 1000)),
        )
        df = fetcher.fetch_backwards(candle_cfg, end_ms=end_ms, target_bars=target_bars)

        rep = validate_ohlcv(df, tf)
        print(
            f"[{sym}] ok={rep.ok} rows={rep.n_rows} dupes={rep.n_dupes} gaps={rep.n_gaps}"
        )

        store.write_partitioned(df, symbol=sym, granularity=tf)


if __name__ == "__main__":
    main()
