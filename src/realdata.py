"""Real financial data loading and rolling-window construction.

Downloads daily prices via yfinance, computes absolute log-returns,
and builds rolling windows in the same dict format as synthetic data.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_returns(tickers, start, end, cache_dir="outputs/data"):
    """Download daily absolute log-returns via yfinance, cache to CSV.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance tickers (e.g. ["^GSPC"]).
    start, end : str
        Date range in "YYYY-MM-DD" format.
    cache_dir : str
        Directory for cached CSV files.

    Returns
    -------
    dict[str, DataFrame]
        Keyed by ticker. Each DataFrame has columns ['date', 'abs_return'].
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    result = {}

    for ticker in tickers:
        safe_name = ticker.replace("^", "").replace("/", "_")
        csv_path = os.path.join(cache_dir, f"returns_{safe_name}.csv")

        if os.path.exists(csv_path):
            logger.info("Loading cached returns for %s from %s", ticker, csv_path)
            df = pd.read_csv(csv_path, parse_dates=["date"])
        else:
            logger.info("Downloading %s from %s to %s …", ticker, start, end)
            data = yf.download(ticker, start=start, end=end, auto_adjust=True,
                               progress=False)
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            close = data["Close"].dropna()
            log_ret = np.log(close / close.shift(1)).dropna()
            abs_ret = log_ret.abs()

            df = pd.DataFrame({
                "date": abs_ret.index,
                "abs_return": abs_ret.values,
            }).reset_index(drop=True)

            df.to_csv(csv_path, index=False)
            logger.info("  → %d returns saved to %s", len(df), csv_path)

        result[ticker] = df

    return result


def rolling_windows(abs_returns, dates, window_size, step_size, ticker):
    """Create rolling windows from a time series.

    Parameters
    ----------
    abs_returns : ndarray
        Absolute log-returns.
    dates : ndarray or Series
        Corresponding dates.
    window_size : int
        Number of observations per window.
    step_size : int
        Step between consecutive windows.
    ticker : str
        Ticker symbol for metadata.

    Returns
    -------
    list[dict]
        Each dict has keys: samples, n, dist_type, params, start_date,
        end_date, window_idx, ticker, full_series_returns, full_series_dates.
    """
    abs_returns = np.asarray(abs_returns, dtype=np.float64)
    dates = np.asarray(dates)
    n_obs = len(abs_returns)
    windows = []

    for idx, start in enumerate(range(0, n_obs - window_size + 1, step_size)):
        end = start + window_size
        window_samples = abs_returns[start:end].copy()

        windows.append({
            "samples": window_samples,
            "n": window_size,
            "dist_type": "real",
            "params": {},
            "start_date": str(dates[start]),
            "end_date": str(dates[end - 1]),
            "window_idx": idx,
            "ticker": ticker,
            # Store indices into the full series for future-returns lookup
            "series_end_idx": end,
        })

    return windows


def prepare_real_datasets(config, cache_dir="outputs/data"):
    """Top-level function: load returns, build rolling windows.

    Parameters
    ----------
    config : dict
        Must contain 'realdata' section with tickers, start, end,
        window_size, step_size.
    cache_dir : str
        Directory for caching.

    Returns
    -------
    datasets : list[dict]
        Rolling-window datasets in the same format as synthetic.
    returns_lookup : dict[str, dict]
        Per-ticker dict with 'abs_returns' (ndarray) and 'dates' (ndarray)
        for future-returns extraction during evaluation.
    """
    rc = config["realdata"]
    tickers = rc["tickers"]
    start = rc["start"]
    end = rc["end"]
    window_size = rc["window_size"]
    step_size = rc["step_size"]

    ticker_data = load_returns(tickers, start, end, cache_dir)

    datasets = []
    returns_lookup = {}

    for ticker, df in ticker_data.items():
        abs_ret = df["abs_return"].values
        dates = df["date"].values

        returns_lookup[ticker] = {
            "abs_returns": abs_ret,
            "dates": dates,
        }

        windows = rolling_windows(abs_ret, dates, window_size, step_size, ticker)
        datasets.extend(windows)
        logger.info("  %s: %d windows (from %d observations)", ticker, len(windows), len(df))

    logger.info("Total real-data windows: %d", len(datasets))
    return datasets, returns_lookup
