from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.bitget_client import BitgetHttpClient
from src.utils.time import granularity_ms, round_down_ms

_CANDLES_PATH = "/api/v2/mix/market/candles"
_HISTORY_PATH = "/api/v2/mix/market/history-candles"

# From Bitget docs for /api/v2/mix/market/candles:
# 15m -> 52 days, 30m -> 62, 1H -> 83, 2H -> 120, 4H -> 240, 6H -> 360, 1m/3m/5m -> ~1 month
# (We use this only to decide when to switch to history-candles.)
_MAX_DAYS_BY_GRAN_FOR_CANDLES: Dict[str, int] = {
    "1m": 31,
    "3m": 31,
    "5m": 31,
    "15m": 52,
    "30m": 62,
    "1H": 83,
    "2H": 120,
    "4H": 240,
    "6H": 360,
}

# /history-candles note: "maximum time query range is 90 days"
_MAX_RANGE_MS_HISTORY = 90 * 24 * 60 * 60_000


@dataclass(frozen=True)
class CandleFetchConfig:
    symbol: str
    granularity: str = "15m"
    product_type: str = "usdt-futures"
    kline_type: str = "MARKET"  # only used by /candles, ignored by /history-candles
    limit: int = 1000  # used by /candles (max 1000). history-candles will clamp to 200.
    mode: str = "auto"  # "auto" | "candles" | "history"
    sleep_sec: float = 0.0  # rate limiting between requests


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
        Fetch candles by paging backwards.
        - If cfg.mode == "candles": uses /candles (fast, but limited history by granularity).
        - If cfg.mode == "history": uses /history-candles (slow, but can go further back; per-request window <= 90 days, max 200 rows).
        - If cfg.mode == "auto": uses /candles if requested span is within its granularity history limit, else uses /history-candles.
        """
        interval = granularity_ms(cfg.granularity)
        end_ms = round_down_ms(end_ms, interval)

        # Decide endpoint
        mode = str(cfg.mode).lower()
        if mode not in ("auto", "candles", "history"):
            raise ValueError(f"Unsupported mode: {cfg.mode}")

        use_history = False
        if mode == "history":
            use_history = True
        elif mode == "candles":
            use_history = False
        else:
            # auto: if you are requesting more than /candles can provide -> history
            max_days = _MAX_DAYS_BY_GRAN_FOR_CANDLES.get(cfg.granularity)
            if max_days is None:
                # unknown granularity -> be conservative
                use_history = True
            else:
                max_bars_candles = int((max_days * 24 * 60 * 60_000) // interval)
                use_history = target_bars > max_bars_candles

        if use_history:
            return self._fetch_backwards_history(
                cfg, end_ms=end_ms, target_bars=target_bars, start_ms=start_ms
            )
        return self._fetch_backwards_candles(
            cfg, end_ms=end_ms, target_bars=target_bars, start_ms=start_ms
        )

    def _fetch_backwards_candles(
        self,
        cfg: CandleFetchConfig,
        end_ms: int,
        target_bars: int,
        start_ms: Optional[int],
    ) -> pd.DataFrame:
        interval = granularity_ms(cfg.granularity)

        all_rows: List[list[str]] = []
        cur_end = end_ms

        while len(all_rows) < target_bars:
            params: Dict[str, Any] = {
                "symbol": cfg.symbol,
                "granularity": cfg.granularity,
                "productType": cfg.product_type,
                "kLineType": cfg.kline_type,
                "limit": str(int(cfg.limit)),
                "endTime": str(cur_end),
            }
            if start_ms is not None:
                params["startTime"] = str(round_down_ms(start_ms, interval))

            resp = self.client.get(_CANDLES_PATH, params=params)
            data = resp.get("data", [])
            if not data:
                break

            all_rows.extend(data)

            # move end backward: earliest timestamp - interval
            earliest = int(min(row[0] for row in data))
            next_end = earliest - interval
            if next_end >= cur_end:
                break
            cur_end = next_end

            if start_ms is not None and earliest <= start_ms:
                break

            if cfg.sleep_sec and cfg.sleep_sec > 0:
                time.sleep(cfg.sleep_sec)

        return self._finalize_rows(all_rows, cfg, target_bars, start_ms)

    def _fetch_backwards_history(
        self,
        cfg: CandleFetchConfig,
        end_ms: int,
        target_bars: int,
        start_ms: Optional[int],
    ) -> pd.DataFrame:
        """
        Uses /api/v2/mix/market/history-candles
        - max 200 rows per call
        - time span between startTime and endTime must be <= 90 days
        """
        interval = granularity_ms(cfg.granularity)
        limit = min(int(cfg.limit), 200)  # history-candles max 200

        # If caller doesn't provide start_ms, infer a target start from target_bars
        if start_ms is None:
            start_ms = end_ms - (target_bars * interval)

        start_ms = round_down_ms(start_ms, interval)

        all_rows: List[list[str]] = []
        cur_end = end_ms

        while len(all_rows) < target_bars and cur_end > start_ms:
            # Keep request window within 90 days
            window_start = max(start_ms, cur_end - _MAX_RANGE_MS_HISTORY)
            window_start = round_down_ms(window_start, interval)

            params: Dict[str, Any] = {
                "symbol": cfg.symbol,
                "granularity": cfg.granularity,
                "productType": cfg.product_type,
                "limit": str(limit),
                "startTime": str(window_start),
                "endTime": str(cur_end),
            }

            resp = self.client.get(_HISTORY_PATH, params=params)
            data = resp.get("data", [])
            if not data:
                # No data in this window: push the window back by 90 days and continue a few times.
                # This handles gaps / listing dates more gracefully than "break immediately".
                cur_end = window_start - interval
                if cfg.sleep_sec and cfg.sleep_sec > 0:
                    time.sleep(cfg.sleep_sec)
                continue

            all_rows.extend(data)

            earliest = int(min(row[0] for row in data))
            next_end = earliest - interval
            if next_end >= cur_end:
                break
            cur_end = next_end

            if cfg.sleep_sec and cfg.sleep_sec > 0:
                time.sleep(cfg.sleep_sec)

        return self._finalize_rows(all_rows, cfg, target_bars, start_ms)

    def _finalize_rows(
        self,
        rows: List[list[str]],
        cfg: CandleFetchConfig,
        target_bars: int,
        start_ms: Optional[int],
    ) -> pd.DataFrame:
        df = self._to_df(rows, cfg)

        df = (
            df.drop_duplicates(subset=["timestamp_ms"])
            .sort_values("timestamp_ms")
            .reset_index(drop=True)
        )

        if start_ms is not None and not df.empty:
            df = df[df["timestamp_ms"] >= start_ms].reset_index(drop=True)

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
