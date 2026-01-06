from __future__ import annotations

import time
import random
import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BitgetClientConfig:
    base_url: str = "https://api.bitget.com"
    timeout_sec: int = 20
    max_retries: int = 6
    backoff_base_sec: float = 0.8
    backoff_jitter_sec: float = 0.3


class BitgetHttpClient:
    def __init__(self, cfg: BitgetClientConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
                # 429 / 5xx: retry
                if r.status_code in (429, 500, 502, 503, 504):
                    self._sleep_backoff(attempt, r.status_code)
                    continue

                r.raise_for_status()
                data = r.json()
                # Bitget uses code "00000" for success on many endpoints
                if isinstance(data, dict) and data.get("code") not in (None, "00000"):
                    raise RuntimeError(f"Bitget API error: {data}")
                return data

            except Exception as e:
                last_err = e
                self._sleep_backoff(attempt, None)

        raise RuntimeError(f"Bitget request failed after retries: {last_err}")

    def _sleep_backoff(self, attempt: int, status_code: Optional[int]) -> None:
        # exponential backoff + jitter (simple + effective)
        base = self.cfg.backoff_base_sec * (2**attempt)
        jitter = random.random() * self.cfg.backoff_jitter_sec
        time.sleep(base + jitter)
