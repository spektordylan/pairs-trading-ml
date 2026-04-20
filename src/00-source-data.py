"""
data_sourcing.py

Retrieves and cleans historical OHLCV data for either equities (via yfinance)
or crypto (via ccxt). Outputs a cleaned DataFrame of log returns.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import urllib.request
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EQUITY_START = "2016-01-01"
EQUITY_END = "2026-01-01"
MISSING_THRESH = 0.95   # drop tickers missing more than 5% of trading days
CRYPTO_LIMIT = 1000   # candles per ccxt request
CCXT_SLEEP = 0.5    # seconds between requests to avoid rate limiting

# ---------------------------------------------------------------------------
# Equities
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })
    with urllib.request.urlopen(req) as response:
        html = response.read()
    table = pd.read_html(html)[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers

def fetch_equity_prices(
    tickers: list[str],
    start: str = EQUITY_START,
    end: str = EQUITY_END,
    ) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers via yfinance.

    Returns:
        prices: DataFrame of shape (days, tickers), adjusted close prices.
    """
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    return prices


def clean_prices(prices: pd.DataFrame, thresh: float = MISSING_THRESH) -> pd.DataFrame:
    """
    Drop tickers with excessive missing data, then forward-fill remaining gaps.

    Args:
        prices:  Raw price DataFrame.
        thresh:  Minimum fraction of non-null rows required to keep a ticker.

    Returns:
        Cleaned price DataFrame.
    """
    min_obs = int(thresh * len(prices))
    prices = prices.dropna(thresh=min_obs, axis=1)
    prices = prices.ffill().dropna()
    return prices


def get_equity_returns(
    tickers: list[str] | None = None,
    start: str = EQUITY_START,
    end: str = EQUITY_END,
    save_path: str | None = "./data/raw/equity_returns.csv"
    ) -> pd.DataFrame:
    """
    Fetch, clean, and compute log returns for equities.

    Args:
        tickers: List of ticker strings. Defaults to current S&P 500.
        start:   Start date string (YYYY-MM-DD).
        end:     End date string (YYYY-MM-DD).

    Returns:
        log_returns: DataFrame of shape (days, tickers).
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    prices      = fetch_equity_prices(tickers, start, end)
    prices      = clean_prices(prices)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        log_returns.to_csv(save_path)
        print(f"[equities] saved to {save_path}")
    
    return log_returns


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------

def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """
    Instantiate a ccxt exchange object.

    Args:
        exchange_id: Any exchange supported by ccxt, e.g. 'binance', 'kraken'.

    Returns:
        Configured ccxt Exchange instance.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,   # respect exchange rate limits automatically
    })
    return exchange


def get_top_crypto_symbols(
    exchange: ccxt.Exchange,
    quote: str = "USDT",
    top_n: int = 100,
    ) -> list[str]:
    """
    Fetch the top N crypto symbols by 24h volume on a given exchange.

    Args:
        exchange: ccxt Exchange instance.
        quote:    Quote currency to filter by (e.g. 'USDT', 'USD').
        top_n:    Number of symbols to return.

    Returns:
        List of symbol strings, e.g. ['BTC/USDT', 'ETH/USDT', ...].
    """
    tickers = exchange.fetch_tickers()

    # Filter to symbols ending in the desired quote currency
    filtered = {
        symbol: data
        for symbol, data in tickers.items()
        if symbol.endswith(f"/{quote}") and data.get("quoteVolume")
    }

    # Sort by 24h quote volume descending
    sorted_symbols = sorted(
        filtered.items(),
        key=lambda x: x[1]["quoteVolume"],
        reverse=True,
    )

    return [sym for sym, _ in sorted_symbols[:top_n]]


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1d",
    since_ms: int | None = None,
    limit: int = CRYPTO_LIMIT,
    ) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a single symbol from a ccxt exchange.

    Args:
        exchange:   ccxt Exchange instance.
        symbol:     Symbol string, e.g. 'BTC/USDT'.
        timeframe:  Candle interval, e.g. '1d', '4h', '1h'.
        since_ms:   Start timestamp in milliseconds (Unix epoch).
        limit:      Max candles per request.

    Returns:
        DataFrame with columns [open, high, low, close, volume],
        indexed by UTC datetime.
    """
    raw = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        since=since_ms,
        limit=limit,
    )

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_crypto_prices(
    symbols: list[str],
    exchange: ccxt.Exchange,
    timeframe: str = "1d",
    start: str = "2024-01-01",
    end: str = "2026-01-01",
    ) -> pd.DataFrame:
    """
    Fetch closing prices for a list of crypto symbols, with basic rate-limit handling.

    Args:
        symbols:   List of ccxt symbol strings.
        exchange:  ccxt Exchange instance.
        timeframe: Candle interval.
        start:     Start date string (YYYY-MM-DD).
        end:       End date string (YYYY-MM-DD).

    Returns:
        DataFrame of close prices, shape (days, symbols).
    """
    since_ms  = exchange.parse8601(f"{start}T00:00:00Z")
    end_dt    = pd.Timestamp(end, tz="UTC")
    close_map = {}

    for symbol in symbols:
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, since_ms)
            if df.empty:
                continue
            df = df[df.index <= end_dt]
            close_map[symbol] = df["close"]
            time.sleep(CCXT_SLEEP)
        except ccxt.BaseError as e:
            print(f"[ccxt] skipping {symbol}: {e}")
            continue

    prices = pd.DataFrame(close_map)
    return prices


def get_crypto_returns(
    symbols: list[str] | None = None,
    exchange_id: str = "kraken",
    timeframe: str = "1d",
    start: str = "2024-01-01",
    end: str = "2026-01-01",
    top_n: int = 100,
    save_path: str | None = "./data/raw/crypto_returns.csv",
    ) -> pd.DataFrame:
    """
    Fetch, clean, and compute log returns for crypto.

    Args:
        symbols:     Explicit list of ccxt symbols. If None, fetches top N by volume.
        exchange_id: ccxt exchange identifier.
        timeframe:   Candle interval.
        start:       Start date string (YYYY-MM-DD).
        end:         End date string (YYYY-MM-DD).
        top_n:       Number of top-volume symbols to use if symbols is None.

    Returns:
        log_returns: DataFrame of shape (days, symbols).
    """
    exchange = get_exchange(exchange_id)

    if symbols is None:
        # hardcode symbols due to issues fetching tickers
        symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD",
        "DOT/USD", "LINK/USD", "AVAX/USD", "DOGE/USD", "LTC/USD",
        "ALGO/USD", "ATOM/USD", "XLM/USD", "TRX/USD", "COMP/USD",
        "NEAR/USD", "UNI/USD", "XMR/USD", "ZEC/USD", "DASH/USD"]

    prices      = fetch_crypto_prices(symbols, exchange, timeframe, start, end)
    prices      = clean_prices(prices)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        log_returns.to_csv(save_path)
        print(f"[crypto] saved to {save_path}")

    return log_returns



if __name__ == "__main__":
    print("=== Equity returns ===")
    eq_returns = get_equity_returns()
    print(eq_returns.head())

    print("\n=== Crypto returns ===")
    cr_returns = get_crypto_returns(top_n=50)
    print(cr_returns.head())