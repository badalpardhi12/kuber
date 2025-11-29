"""Centralized market data provider helpers."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "BcdfA3POzO0fizDHmDmtFsGpExZ_biSX")
_POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/minute/{start}/{end}"
_CACHE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".cache" / "polygon"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MAX_RETRIES = 4
_RETRY_BACKOFF_SECONDS = 2.0


def has_polygon_api_key() -> bool:
    """Return True if a Polygon API key is available."""
    return bool(_POLYGON_API_KEY)


def _cache_path(symbol: str, start: datetime, end: datetime, interval: int) -> Path:
    filename = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{interval}m.pkl"
    return _CACHE_DIR / filename


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(path)
    except Exception:
        pass


def _polygon_request(symbol: str, multiplier: int, start: datetime, end: datetime) -> Dict:
    url = _POLYGON_BASE.format(
        symbol=symbol.upper(),
        multiplier=multiplier,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": _POLYGON_API_KEY,
    }

    delay = _RETRY_BACKOFF_SECONDS
    for attempt in range(_MAX_RETRIES):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt == _MAX_RETRIES - 1:
                response.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()

    # Should never reach here, but keep mypy happy
    response.raise_for_status()
    return {}


def _results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    rename_map = {
        "t": "datetime",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)
    df["datetime"] = (
        pd.to_datetime(df["datetime"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
    )
    df["date"] = df["datetime"].dt.date
    cols = ["datetime", "open", "high", "low", "close", "volume", "date"]
    return df[cols]


def fetch_polygon_intraday(
    symbol: str,
    start: datetime,
    end: datetime,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Fetch intraday bars for a symbol from Polygon."""
    if not has_polygon_api_key():
        raise RuntimeError("POLYGON_API_KEY is not configured")
    if start >= end:
        raise ValueError("start must be before end")
    cache_path = _cache_path(symbol, start, end, interval_minutes)
    cached = _read_cache(cache_path)
    if not cached.empty:
        return cached

    data = _polygon_request(symbol, interval_minutes, start, end)
    results = data.get("results", [])
    df = _results_to_dataframe(results)
    if not df.empty:
        _write_cache(cache_path, df)
    return df


def download_polygon_intraday_history(
    symbols: List[str],
    start: datetime,
    end: datetime,
    interval_minutes: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Download intraday history for multiple symbols."""
    datasets: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = fetch_polygon_intraday(symbol, start, end, interval_minutes)
            if df.empty:
                print(f"  Downloading {symbol} intraday... ❌ no data")
            else:
                print(f"  Downloading {symbol} intraday... ✅ {len(df)} bars")
            datasets[symbol] = df
        except requests.HTTPError as exc:
            print(f"  Downloading {symbol} intraday... ❌ HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            print(f"  Downloading {symbol} intraday... ❌ {exc}")
        except RuntimeError as exc:
            print(f"  Downloading {symbol} intraday... ❌ {exc}")
    return datasets
