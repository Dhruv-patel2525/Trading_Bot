from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.utils.time import ms_to_ym


@dataclass(frozen=True)
class ParquetStoreConfig:
    root: Path  # e.g. data/raw
    source: str  # "bitget"
    product_type: str  # "USDT-FUTURES" or "usdt-futures" (use one consistently)


class ParquetStore:
    def __init__(self, cfg: ParquetStoreConfig):
        self.cfg = cfg

    def dataset_path(self, symbol: str, granularity: str) -> Path:
        return (
            self.cfg.root
            / self.cfg.source
            / self.cfg.product_type
            / symbol
            / granularity
        )

    def write_partitioned(
        self, df: pd.DataFrame, symbol: str, granularity: str
    ) -> Path:
        """
        Write as partitioned parquet (year/month) under:
        data/raw/<source>/<productType>/<symbol>/<granularity>/year=YYYY/month=MM/part-*.parquet
        """
        if df.empty:
            return self.dataset_path(symbol, granularity)

        years, months = [], []
        for ts in df["timestamp_ms"].tolist():
            y, m = ms_to_ym(int(ts))
            years.append(y)
            months.append(m)

        dfx = df.copy()
        dfx["year"] = years
        dfx["month"] = months

        base = self.dataset_path(symbol, granularity)
        base.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(dfx, preserve_index=False)
        ds.write_dataset(
            data=table,
            base_dir=str(base),
            format="parquet",
            partitioning=["year", "month"],
            existing_data_behavior="overwrite_or_ignore",  # safe for incremental writes
        )
        return base

    def read_dataset(self, symbol: str, granularity: str) -> pd.DataFrame:
        base = self.dataset_path(symbol, granularity)
        if not base.exists():
            return pd.DataFrame()
        dataset = ds.dataset(str(base), format="parquet", partitioning="hive")
        table = dataset.to_table()
        return table.to_pandas()
