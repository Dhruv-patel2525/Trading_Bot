from __future__ import annotations

import argparse
from pathlib import Path

from src.data.bitget_client import BitgetClientConfig, BitgetHttpClient
from src.data.bitget_candles import CandleFetchConfig, BitgetCandleFetcher
from src.data.store import ParquetStore, ParquetStoreConfig
from src.data.validate import validate_ohlcv
from src.utils.time import utcnow_ms, granularity_ms


# to run the script:
# uv run python -m src.data.fetch_bitget --symbol BTCUSDT --tf 15m --days 7
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--tf", default="15m", help="e.g. 15m")
    ap.add_argument("--days", type=int, default=7, help="How many days back to fetch")
    ap.add_argument("--product-type", default="usdt-futures", help="e.g. usdt-futures")
    ap.add_argument("--out-root", default="data/raw", help="Raw data root folder")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    interval = granularity_ms(args.tf)
    end_ms = utcnow_ms()
    target_bars = int((args.days * 24 * 60 * 60_000) // interval)

    client = BitgetHttpClient(BitgetClientConfig())
    fetcher = BitgetCandleFetcher(client)

    cfg = CandleFetchConfig(
        symbol=args.symbol,
        granularity=args.tf,
        product_type=args.product_type,
        limit=args.limit,
    )

    df = fetcher.fetch_backwards(cfg, end_ms=end_ms, target_bars=target_bars)

    report = validate_ohlcv(df, args.tf)
    print(
        f"[validate] ok={report.ok} rows={report.n_rows} dupes={report.n_dupes} gaps={report.n_gaps} "
        f"first={report.first_ts} last={report.last_ts}"
    )

    store = ParquetStore(
        ParquetStoreConfig(
            root=Path(args.out_root),
            source="bitget",
            product_type=args.product_type,
        )
    )

    path = store.write_partitioned(df, symbol=args.symbol, granularity=args.tf)
    print(f"[write] wrote dataset to: {path}")


if __name__ == "__main__":
    main()
