"""Centralized market data provider helpers."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "BcdfA3POzO0fizDHmDmtFsGpExZ_biSX")
_POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/minute/{start}/{end}"
_CACHE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".cache" / "polygon"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MAX_RETRIES = 10
_RETRY_BACKOFF_SECONDS = 5.0
_CHUNK_DAYS = int(os.environ.get("POLYGON_MAX_CHUNK_DAYS", 90))
_THROTTLE_BETWEEN_SYMBOLS = float(os.environ.get("POLYGON_SYMBOL_THROTTLE_SECONDS", 1.0))


def has_polygon_api_key() -> bool:
    """Return True if a Polygon API key is available."""
    return bool(_POLYGON_API_KEY)


def _cache_path(symbol: str, start: datetime, end: datetime, interval: int) -> Path:
    filename = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{interval}m.pkl"
    return _CACHE_DIR / filename


@contextmanager
def _cache_lock(path: Path):
    lock_path = Path(f"{path}.lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.25)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


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


class RateLimiter:
    """Simple rate limiter for Polygon free tier (5 req / minute)."""
    def __init__(self, max_per_minute: int = 5):
        self.delay = 60.0 / max_per_minute
        self.last_req_time = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_req_time
        if elapsed < self.delay:
            to_sleep = self.delay - elapsed
            # Add a tiny bit of jitter
            time.sleep(to_sleep + 0.1)
        self.last_req_time = time.time()

_RATE_LIMITER = RateLimiter(max_per_minute=5)

def _polygon_request(
    symbol: str,
    multiplier: int,
    start: datetime,
    end: datetime,
    next_url: Optional[str] = None,
) -> Dict:
    if next_url is None:
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
    else:
        url = next_url
        params = {}
        if "apiKey=" not in url:
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}apiKey={_POLYGON_API_KEY}"

    # Proactive rate limiting
    _RATE_LIMITER.wait()

    delay = _RETRY_BACKOFF_SECONDS
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == _MAX_RETRIES - 1:
                    response.raise_for_status()
                
                # Check for specific Retry-After header
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait_time = float(retry_after)
                else:
                    # Exponential backoff with jitter
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 1.0)
                
                print(f"⚠️ Polygon {response.status_code} for {symbol}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
             if attempt == _MAX_RETRIES - 1:
                 raise e
             print(f"⚠️ Request error for {symbol}: {e}. Retrying...")
             time.sleep(delay)
             continue

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
    frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_date_ranges(start, end):
        chunk_df = _fetch_chunk(symbol, chunk_start, chunk_end, interval_minutes)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df.reset_index(drop=True, inplace=True)
    return df


def _chunk_date_ranges(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    if _CHUNK_DAYS <= 0:
        return [(start, end)]
    ranges: List[Tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=_CHUNK_DAYS), end)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end
    return ranges


def _fetch_chunk(symbol: str, start: datetime, end: datetime, interval_minutes: int) -> pd.DataFrame:
    cache_path = _cache_path(symbol, start, end, interval_minutes)
    cached = _read_cache(cache_path)
    if not cached.empty:
        return cached

    with _cache_lock(cache_path):
        cached = _read_cache(cache_path)
        if not cached.empty:
            return cached

        results: List[Dict] = []
        next_url: Optional[str] = None

        while True:
            data = _polygon_request(symbol, interval_minutes, start, end, next_url=next_url)
            results.extend(data.get("results", []))
            next_url = data.get("next_url")
            if not next_url:
                break

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
        finally:
            if _THROTTLE_BETWEEN_SYMBOLS > 0:
                time.sleep(_THROTTLE_BETWEEN_SYMBOLS)
    return datasets
