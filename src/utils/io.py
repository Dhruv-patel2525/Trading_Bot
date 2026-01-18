"""I/O helpers: config loading, path helpers, hashing configs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_parquet_any(path: str | Path) -> pd.DataFrame:
    """
    Reads either:
      - a single parquet file
      - a directory of partitioned parquet files (hive-style)
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    if path.is_file():
        return pd.read_parquet(path)

    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    table = dataset.to_table()
    return table.to_pandas()


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path
