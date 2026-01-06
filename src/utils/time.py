from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

_GRAN_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1H": 60 * 60_000,
    "4H": 4 * 60 * 60_000,
    "6H": 6 * 60 * 60_000,
    "12H": 12 * 60 * 60_000,
    "1D": 24 * 60 * 60_000,
    "3D": 3 * 24 * 60 * 60_000,
    "1W": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,  # approx for rounding only
    # utc variants if you ever use them
    "6Hutc": 6 * 60 * 60_000,
    "12Hutc": 12 * 60 * 60_000,
    "1Dutc": 24 * 60 * 60_000,
    "3Dutc": 3 * 24 * 60 * 60_000,
    "1Wutc": 7 * 24 * 60 * 60_000,
    "1Mutc": 30 * 24 * 60 * 60_000,
}


def granularity_ms(granularity: str) -> int:
    if granularity not in _GRAN_MS:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return _GRAN_MS[granularity]


def utcnow_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def round_down_ms(ts_ms: int, interval_ms: int) -> int:
    return ts_ms - (ts_ms % interval_ms)


def ms_to_ym(ts_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.year, dt.month
