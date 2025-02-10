import numpy as np
import pandas as pd


def ewma_beta(series, period, beta=1.0):
    """
    Compute Exponentially Weighted Moving Average (EWMA) with a beta-modified alpha factor.

    Parameters:
    - series: Pandas Series of prices.
    - period: Lookback period.
    - beta: Adjusts the smoothing factor.

    Returns:
    - EWMA Series.
    """
    alpha = beta * (2 / (period + 1))
    return series.ewm(alpha=alpha, adjust=False).mean()


def weighted_moving_average(series, period):
    """
    Compute Weighted Moving Average (WMA).

    Parameters:
    - series: Pandas Series of prices.
    - period: Lookback period for WMA.

    Returns:
    - Series of WMA values.
    """
    weights = np.arange(1, period + 1)  # Linear weights (1,2,3,...,N)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hull_moving_average(price_series, length):
    """
    Compute the Hull Moving Average (HMA).

    Parameters:
    - price_series: Pandas Series of prices.
    - length: Lookback period for the HMA.

    Returns:
    - Series of Hull Moving Average values.
    """
    if length < 1:
        raise ValueError("Length must be greater than 0.")

    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    wma_half = weighted_moving_average(price_series, half_length)
    wma_full = weighted_moving_average(price_series, length)
    hma = weighted_moving_average(2 * wma_half - wma_full, sqrt_length)

    return hma


def adaptive_hull_moving_average(price_series, period, beta=1.0):
    """
    Compute an Adaptive Hull Moving Average (AHMA) using EWMA instead of WMA.

    Parameters:
    - price_series: Pandas Series of prices.
    - period: Lookback period for adaptive smoothing.
    - beta: Adjusts the smoothing factor.

    Returns:
    - Series of Adaptive HMA values.
    """
    if period < 1:
        raise ValueError("Period must be greater than 0.")

    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    ewma_half = ewma_beta(price_series, half_length, beta=beta)
    ewma_full = ewma_beta(price_series, period, beta=beta)
    ahma = ewma_beta(2 * ewma_half - ewma_full, sqrt_length, beta=beta)

    return pd.Series(ahma, index=price_series.index)


def linreg_features(price_series, length, tgt_bar=0):
    """
    Compute Linear Regression projected values, along with slope and intercept.

    Parameters:
    - price_series (pd.Series): Price data (e.g., Close).
    - length (int): Lookback period for regression.
    - tgt_bar (int): Offset bar (-N for future projection, +N for past projection).

    Returns:
    - tuple:
        - pd.Series: Projected regression values.
        - pd.Series: Slopes (rate of change).
        - pd.Series: Intercepts (baseline at x=0).
    """
    if length < 2:
        raise ValueError("Length must be at least 2 for regression calculation.")

    source = np.asarray(price_series, dtype=np.float64)
    linreg_values = np.full_like(source, np.nan)
    slopes = np.full_like(source, np.nan)
    intercepts = np.full_like(source, np.nan)

    for i in range(length - 1, len(source)):
        X = np.arange(length).reshape(-1, 1)
        y = source[i - length + 1 : i + 1].reshape(-1, 1)

        # Compute linear regression coefficients (intercept & slope)
        X_b = np.c_[np.ones((length, 1)), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        intercept, slope = theta.flatten()
        linreg_values[i] = intercept + slope * (length - 1 - tgt_bar)
        slopes[i] = slope
        intercepts[i] = intercept

    return (
        pd.Series(linreg_values, index=price_series.index).bfill(),
        pd.Series(slopes, index=price_series.index),
        pd.Series(intercepts, index=price_series.index),
    )


def leavitt_projection(price_series, length):
    """
    Compute the Leavitt Projection (1-bar forward forecast using Linear Regression).

    Parameters:
    - price_series (pd.Series): Series of prices.
    - length (int): Lookback period for regression.

    Returns:
    - pd.Series: Leavitt Projection values.
    """
    return linreg_features(price_series, length, tgt_bar=-1)[0]


def leavitt_convolution(price_series, plength, clength):
    """
    Compute the Leavitt Convolution, which smooths the Leavitt Projection.

    Parameters:
    - price_series (pd.Series): Series of prices.
    - plength (int): Lookback period for Leavitt Projection.
    - clength (int): Lookback period for Leavitt Convolution.

    Returns:
    - tuple:
        - pd.Series: Leavitt Convolution values.
        - pd.Series: Slopes of the Convolution.
        - pd.Series: Intercepts of the Convolution.
    """
    leavitt_proj = leavitt_projection(price_series, plength)
    return linreg_features(leavitt_proj, clength, tgt_bar=-1)


def leavitt_acceleration(lc_slope_series):
    """
    Compute the Leavitt Acceleration (LCACCELERATION),
    which measures the rate of change of the slope of Leavitt Convolution.

    Parameters:
    - lc_slope_series: Pandas Series of Leavitt Convolution Slope values.

    Returns:
    - Pandas Series of Acceleration values.
    """
    return lc_slope_series.diff()


def convolution_probability(lc_slope_series, lc_intercept_series, window=10):
    """
    Compute Convolution Probability Function for market turning points.

    Parameters:
    - lc_slope_series (pd.Series): Leavitt Convolution Slope values.
    - lc_intercept_series (pd.Series): Leavitt Convolution Intercept values.
    - window (int): Rolling window for standard deviation calculation.

    Returns:
    - pd.Series: Probability values.
    """
    slope_std = lc_slope_series.rolling(window=window).std()
    intercept_std = lc_intercept_series.rolling(window=window).std()

    prob_slope = (lc_slope_series - lc_slope_series.mean()) / (slope_std + 1e-6)
    prob_intercept = (lc_intercept_series - lc_intercept_series.mean()) / (
        intercept_std + 1e-6
    )

    return prob_slope * (1 - prob_intercept)
