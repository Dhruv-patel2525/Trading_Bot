from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd

from src.data.bitget_client import BitgetHttpClient
from src.utils.time import granularity_ms, round_down_ms

_CANDLES_PATH = "/api/v2/mix/market/candles"


@dataclass(frozen=True)
class CandleFetchConfig:
    symbol: str
    granularity: str = "15m"
    product_type: str = (
        "usdt-futures"  # Bitget examples often use lowercase with hyphen
    )
    kline_type: str = "MARKET"
    limit: int = 1000


class BitgetCandleFetcher:
    def __init__(self, client: BitgetHttpClient):
        self.client = client

    def fetch_backwards(
        self,
        cfg: CandleFetchConfig,
        end_ms: int,
        target_bars: int,
        start_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch candles by paging backwards using endTime+limit.
        Stops when we collected target_bars or reached start_ms (if provided).
        """
        interval = granularity_ms(cfg.granularity)
        end_ms = round_down_ms(end_ms, interval)

        all_rows: List[list[str]] = []
        cur_end = end_ms

        while len(all_rows) < target_bars:
            params: Dict[str, Any] = {
                "symbol": cfg.symbol,
                "granularity": cfg.granularity,
                "productType": cfg.product_type,
                "kLineType": cfg.kline_type,
                "limit": str(cfg.limit),
                "endTime": str(cur_end),
            }
            if start_ms is not None:
                params["startTime"] = str(round_down_ms(start_ms, interval))

            resp = self.client.get(_CANDLES_PATH, params=params)
            data = resp.get("data", [])
            if not data:
                break

            # data is list[list[str]] => [ts, open, high, low, close, vol_base, vol_quote]
            all_rows.extend(data)

            # move end backward: earliest timestamp - interval
            try:
                earliest = int(min(row[0] for row in data))
            except Exception:
                break

            next_end = earliest - interval
            if next_end >= cur_end:
                # safety to avoid infinite loops if API returns weird ordering
                break

            cur_end = next_end

            if start_ms is not None and earliest <= start_ms:
                break

        df = self._to_df(all_rows, cfg)
        # dedupe + sort
        df = (
            df.drop_duplicates(subset=["timestamp_ms"])
            .sort_values("timestamp_ms")
            .reset_index(drop=True)
        )

        # if start_ms provided, trim
        if start_ms is not None:
            df = df[df["timestamp_ms"] >= start_ms].reset_index(drop=True)

        # if we overshot target bars, keep the most recent target_bars
        if len(df) > target_bars:
            df = df.iloc[-target_bars:].reset_index(drop=True)

        return df

    def _to_df(self, rows: List[list[str]], cfg: CandleFetchConfig) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "timestamp_ms",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume_base",
                    "volume_quote",
                    "symbol",
                    "granularity",
                    "source",
                ]
            )

        out = pd.DataFrame(
            rows,
            columns=[
                "timestamp_ms",
                "open",
                "high",
                "low",
                "close",
                "volume_base",
                "volume_quote",
            ],
        )

        out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
        for c in ["open", "high", "low", "close", "volume_base", "volume_quote"]:
            out[c] = out[c].astype("float64")

        out["symbol"] = cfg.symbol
        out["granularity"] = cfg.granularity
        out["source"] = "bitget"
        return out
