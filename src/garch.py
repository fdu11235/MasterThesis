"""GARCH(1,1) fitting and volatility filtering for GARCH-POT.

Implements the McNeil & Frey (2000) approach: fit GARCH(1,1) to signed
log-returns, extract standardized residuals z_t = r_t / sigma_t, and
forecast conditional volatility sigma_{T+h}.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def fit_garch_and_filter(returns, forecast_horizon=1):
    """Fit GARCH(1,1) to returns, return standardized residuals + volatility forecasts.

    Parameters
    ----------
    returns : ndarray
        Raw (signed) log-returns for the window.
    forecast_horizon : int
        Number of steps ahead to forecast volatility.

    Returns
    -------
    dict with:
        'std_residuals': standardized residuals z_t = r_t / sigma_t
        'abs_std_residuals': |z_t|
        'conditional_vol': sigma_t within the window
        'forecast_vol': sigma_{T+1:T+h} forecasted volatilities (length = forecast_horizon)
        'converged': bool
    """
    from arch import arch_model

    returns = np.asarray(returns, dtype=np.float64)

    # Scale returns to percentage for numerical stability (arch convention)
    scale = 100.0
    returns_pct = returns * scale

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='t',
                            mean='Zero', rescale=False)
            res = am.fit(disp='off', show_warning=False)

        if not res.convergence_flag == 0:
            logger.debug("GARCH did not converge, using fallback")
            return _fallback(returns, forecast_horizon)

        # Conditional volatility (in percentage scale)
        cond_vol_pct = res.conditional_volatility
        cond_vol = cond_vol_pct / scale

        # Standardized residuals
        std_resid = returns / np.maximum(cond_vol, 1e-10)

        # Forecast volatilities
        fcast = res.forecast(horizon=forecast_horizon)
        # fcast.variance has shape (n, horizon), last row is the forecast from T
        forecast_var_pct = fcast.variance.iloc[-1].values  # (horizon,)
        forecast_vol = np.sqrt(forecast_var_pct) / scale

        return {
            'std_residuals': std_resid,
            'abs_std_residuals': np.abs(std_resid),
            'conditional_vol': cond_vol,
            'forecast_vol': forecast_vol,
            'converged': True,
        }

    except Exception as e:
        logger.debug("GARCH fitting failed (%s), using fallback", e)
        return _fallback(returns, forecast_horizon)


def _fallback(returns, forecast_horizon):
    """Fallback when GARCH doesn't converge: use constant volatility."""
    vol = np.std(returns)
    vol = max(vol, 1e-10)
    return {
        'std_residuals': returns / vol,
        'abs_std_residuals': np.abs(returns) / vol,
        'conditional_vol': np.full_like(returns, vol),
        'forecast_vol': np.full(forecast_horizon, vol),
        'converged': False,
    }
