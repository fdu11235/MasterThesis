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
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if "signed_return" not in df.columns:
                logger.info("Cached CSV for %s missing signed_return, re-downloading", ticker)
                os.remove(csv_path)
            else:
                logger.info("Loading cached returns for %s from %s", ticker, csv_path)

        if not os.path.exists(csv_path):
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
                "signed_return": log_ret.values,
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
        signed_ret = df["signed_return"].values if "signed_return" in df.columns else None

        returns_lookup[ticker] = {
            "abs_returns": abs_ret,
            "dates": dates,
            "signed_returns": signed_ret,
        }

        windows = rolling_windows(abs_ret, dates, window_size, step_size, ticker)
        datasets.extend(windows)
        logger.info("  %s: %d windows (from %d observations)", ticker, len(windows), len(df))

    logger.info("Total real-data windows: %d", len(datasets))
    return datasets, returns_lookup


def prepare_real_datasets_garch(config, returns_lookup, datasets, cache_dir="outputs/data"):
    """Build GARCH-filtered rolling windows from existing real datasets.

    For each window, fits GARCH(1,1) to the signed returns within the window,
    then replaces the samples with |standardized residuals| and stores the
    forecasted conditional volatilities for the backtest horizon.

    Parameters
    ----------
    config : dict
        Must contain 'realdata' section with backtest_horizon.
    returns_lookup : dict
        Per-ticker dict with 'signed_returns' and 'abs_returns' arrays.
    datasets : list[dict]
        Original unconditional rolling-window datasets.
    cache_dir : str
        Directory for caching.

    Returns
    -------
    garch_datasets : list[dict]
        GARCH-filtered window datasets. Each has 'samples' = |z_t|,
        'garch_forecast_vol', and 'garch_converged' fields.
    """
    from src.garch import fit_garch_and_filter

    backtest_horizon = config["realdata"]["backtest_horizon"]
    garch_datasets = []
    n_converged = 0

    for ds in datasets:
        ticker = ds["ticker"]
        series_end_idx = ds["series_end_idx"]
        window_size = ds["n"]

        signed_ret = returns_lookup[ticker].get("signed_returns")
        if signed_ret is None:
            logger.warning("No signed returns for %s, skipping GARCH window", ticker)
            continue

        # Extract signed returns for this window
        window_start = series_end_idx - window_size
        window_signed = signed_ret[window_start:series_end_idx]

        if len(window_signed) != window_size:
            continue

        garch_result = fit_garch_and_filter(window_signed,
                                            forecast_horizon=backtest_horizon)

        garch_ds = dict(ds)  # shallow copy
        garch_ds["samples"] = garch_result["abs_std_residuals"].copy()
        garch_ds["garch_forecast_vol"] = garch_result["forecast_vol"]
        garch_ds["garch_converged"] = garch_result["converged"]
        garch_ds["garch_conditional_vol"] = garch_result["conditional_vol"]

        garch_datasets.append(garch_ds)
        if garch_result["converged"]:
            n_converged += 1

    logger.info("GARCH-filtered windows: %d total, %d converged",
                len(garch_datasets), n_converged)
    return garch_datasets


def prepare_real_datasets_signsplit(config, returns_lookup, datasets, tail_mode,
                                    min_obs=60):
    """Build sign-split rolling windows from existing unconditional datasets.

    Parameters
    ----------
    config : dict
        Must contain 'realdata' section.
    returns_lookup : dict
        Per-ticker dict with 'signed_returns' arrays.
    datasets : list[dict]
        Original unconditional rolling-window datasets.
    tail_mode : str
        ``"loss"`` (left tail: -r_t for r_t < 0) or ``"profit"``
        (right tail: r_t for r_t > 0).
    min_obs : int
        Skip windows with fewer than this many filtered observations.

    Returns
    -------
    list[dict]
        Sign-filtered window datasets with positive ``"samples"``.
    """
    if tail_mode not in ("loss", "profit"):
        raise ValueError(f"tail_mode must be 'loss' or 'profit', got {tail_mode!r}")

    split_datasets = []
    n_skipped = 0

    for ds in datasets:
        ticker = ds["ticker"]
        series_end_idx = ds["series_end_idx"]
        window_size = ds["n"]

        signed_ret = returns_lookup[ticker].get("signed_returns")
        if signed_ret is None:
            n_skipped += 1
            continue

        window_start = series_end_idx - window_size
        window_signed = signed_ret[window_start:series_end_idx]

        if len(window_signed) != window_size:
            n_skipped += 1
            continue

        if tail_mode == "loss":
            mask = window_signed < 0
            filtered = -window_signed[mask]  # flip sign → positive loss magnitudes
        else:  # profit
            mask = window_signed > 0
            filtered = window_signed[mask]

        if len(filtered) < min_obs:
            n_skipped += 1
            continue

        split_ds = dict(ds)
        split_ds["samples"] = filtered.copy()
        split_ds["n"] = len(filtered)
        split_ds["tail_mode"] = tail_mode
        split_ds["dist_type"] = "real"
        split_datasets.append(split_ds)

    logger.info("Sign-split (%s): %d windows produced, %d skipped (min_obs=%d)",
                tail_mode, len(split_datasets), n_skipped, min_obs)
    return split_datasets


def prepare_real_datasets_garch_signsplit(config, returns_lookup, datasets,
                                          tail_mode, min_obs=60):
    """Build GARCH-filtered, sign-split rolling windows.

    Fits GARCH(1,1) to the full signed returns (same as unconditional GARCH),
    then splits the standardized residuals by the sign of the original returns.

    Parameters
    ----------
    config : dict
        Must contain 'realdata' section with backtest_horizon.
    returns_lookup : dict
        Per-ticker dict with 'signed_returns' arrays.
    datasets : list[dict]
        Original unconditional rolling-window datasets.
    tail_mode : str
        ``"loss"`` or ``"profit"``.
    min_obs : int
        Skip windows with fewer than this many filtered observations.

    Returns
    -------
    list[dict]
        GARCH-filtered, sign-split window datasets.
    """
    from src.garch import fit_garch_and_filter

    if tail_mode not in ("loss", "profit"):
        raise ValueError(f"tail_mode must be 'loss' or 'profit', got {tail_mode!r}")

    backtest_horizon = config["realdata"]["backtest_horizon"]
    split_datasets = []
    n_skipped = 0
    n_converged = 0

    for ds in datasets:
        ticker = ds["ticker"]
        series_end_idx = ds["series_end_idx"]
        window_size = ds["n"]

        signed_ret = returns_lookup[ticker].get("signed_returns")
        if signed_ret is None:
            n_skipped += 1
            continue

        window_start = series_end_idx - window_size
        window_signed = signed_ret[window_start:series_end_idx]

        if len(window_signed) != window_size:
            n_skipped += 1
            continue

        garch_result = fit_garch_and_filter(window_signed,
                                            forecast_horizon=backtest_horizon)

        # Split standardized residuals by sign of original returns
        std_resid = garch_result["std_residuals"]  # signed z_t
        if tail_mode == "loss":
            mask = window_signed < 0
            filtered = np.abs(std_resid[mask])
        else:  # profit
            mask = window_signed > 0
            filtered = np.abs(std_resid[mask])

        if len(filtered) < min_obs:
            n_skipped += 1
            continue

        split_ds = dict(ds)
        split_ds["samples"] = filtered.copy()
        split_ds["n"] = len(filtered)
        split_ds["tail_mode"] = tail_mode
        split_ds["garch_forecast_vol"] = garch_result["forecast_vol"]
        split_ds["garch_converged"] = garch_result["converged"]
        split_ds["garch_conditional_vol"] = garch_result["conditional_vol"]
        split_ds["dist_type"] = "real"
        split_datasets.append(split_ds)

        if garch_result["converged"]:
            n_converged += 1

    logger.info("GARCH sign-split (%s): %d windows, %d converged, %d skipped",
                tail_mode, len(split_datasets), n_converged, n_skipped)
    return split_datasets
