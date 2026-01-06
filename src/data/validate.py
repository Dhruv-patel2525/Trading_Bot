from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.utils.time import granularity_ms


@dataclass(frozen=True)
class IntegrityReport:
    ok: bool
    n_rows: int
    n_dupes: int
    n_gaps: int
    first_ts: int | None
    last_ts: int | None


def validate_ohlcv(df: pd.DataFrame, granularity: str) -> IntegrityReport:
    if df.empty:
        return IntegrityReport(False, 0, 0, 0, None, None)

    interval = granularity_ms(granularity)

    s = df["timestamp_ms"].sort_values()
    n_dupes = int(s.duplicated().sum())

    diffs = s.diff().dropna()
    # gaps: any diff not equal to interval
    gaps = diffs[diffs != interval]
    n_gaps = int((gaps > interval).sum())  # treat only missing-candle gaps as gaps

    ok = (n_dupes == 0) and (n_gaps == 0)

    return IntegrityReport(
        ok=ok,
        n_rows=int(len(df)),
        n_dupes=n_dupes,
        n_gaps=n_gaps,
        first_ts=int(s.iloc[0]),
        last_ts=int(s.iloc[-1]),
    )
