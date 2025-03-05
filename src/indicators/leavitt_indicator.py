import numpy as np
import pandas as pd

# =========================================
# Moving Averages & Smoothing Functions
# =========================================


def ewma_beta(series, period, beta=1.0):
    """
    Compute an Exponentially Weighted Moving Average (EWMA) with a beta-modified alpha factor.

    Parameters:
    - series (pd.Series): Input time series data (e.g., Close prices).
    - period (int): Lookback period for smoothing.
    - beta (float, optional): Adjusts the smoothing factor, with 1.0 being the standard EWMA. Default is 1.0.

    Returns:
    - pd.Series: EWMA smoothed series.
    """
    alpha = beta * (2 / (period + 1))  # Compute the smoothing factor
    return series.ewm(alpha=alpha, adjust=False).mean()


def weighted_moving_average(series, period):
    """
    Compute a Weighted Moving Average (WMA), where recent values have more weight.

    Parameters:
    - series (pd.Series): Input time series data.
    - period (int): Lookback period for the WMA.

    Returns:
    - pd.Series: Weighted Moving Average series.
    """
    weights = np.arange(1, period + 1)  # Assign increasing weights (1,2,3,...,N)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hull_moving_average(price_series, length):
    """
    Compute the Hull Moving Average (HMA), which reduces lag in moving averages.

    Parameters:
    - price_series (pd.Series): Input price series.
    - length (int): Lookback period for smoothing.

    Returns:
    - pd.Series: Hull Moving Average series.
    """
    if length < 1:
        raise ValueError("Length must be greater than 0.")

    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    # Compute Weighted Moving Averages at different lengths
    wma_half = weighted_moving_average(price_series, half_length)
    wma_full = weighted_moving_average(price_series, length)

    # Final HMA calculation
    hma = weighted_moving_average(2 * wma_half - wma_full, sqrt_length)

    return hma


def adaptive_hull_moving_average(price_series, period, beta=1.0):
    """
    Compute an Adaptive Hull Moving Average (AHMA) using EWMA instead of WMA.

    The Adaptive HMA smooths price movements while reducing lag, replacing
    Weighted Moving Averages (WMA) with Exponentially Weighted Moving Averages (EWMA)
    for dynamic responsiveness.

    Parameters:
    - price_series (pd.Series): Input time series (Close, Open, etc.).
    - period (int): Lookback period for smoothing.
    - beta (float, optional): Adjusts the EWMA smoothing factor. Default is 1.0.

    Returns:
    - pd.Series: Adaptive Hull Moving Average (AHMA) values.
    """
    if period < 1:
        raise ValueError("Period must be greater than 0.")

    half_length = int(period / 2)  # Half the period
    sqrt_length = int(np.sqrt(period))  # Square root of the period

    # Compute EWMA at different lengths
    ewma_half = ewma_beta(price_series, half_length, beta=beta)
    ewma_full = ewma_beta(price_series, period, beta=beta)

    # Final Adaptive HMA calculation
    ahma = ewma_beta(2 * ewma_half - ewma_full, sqrt_length, beta=beta)

    return pd.Series(ahma, index=price_series.index)


# =========================================
# Linear Regression Functions
# =========================================


def ta_linreg(price_series, length, tgt_bar=0, method="lsr"):
    """
    Compute the Linear Regression value similar to TradingView's ta.linreg()
    and TradeStation's LinearRegValue(), allowing for method selection.

    Parameters:
    - price_series (pd.Series): Input time series (Close, Open, etc.).
    - length (int): Lookback period for regression.
    - tgt_bar (int, optional): Offset for projection (-N for future, +N for past). Default is 0.
    - method (str, optional): Regression method selection.
        - "lsr" (Least Squares Regression, uses np.polyfit) [Default]
        - "pinv" (Pseudo-Inverse, uses np.linalg.pinv)

    Returns:
    - pd.Series: Regression values at each bar.
    """
    if length < 2:
        raise ValueError("Length must be at least 2 for regression calculation.")

    source = np.asarray(price_series, dtype=np.float64)
    linreg_values = np.full_like(source, np.nan)

    for i in range(length - 1, len(source)):
        X = np.arange(length).reshape(-1, 1)
        y = source[i - length + 1 : i + 1].reshape(-1, 1)

        if method == "lsr":
            # ✅ Least Squares Regression (Matches TradingView & TradeStation)
            slope, intercept = np.polyfit(X.flatten(), y.flatten(), deg=1)

        elif method == "pinv":
            # ✅ Pseudo-Inverse Approach (Stable for ill-conditioned data)
            X_b = np.c_[np.ones((length, 1)), X]  # Add bias term (x0 = 1)
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            intercept, slope = theta.flatten()

        else:
            raise ValueError("Invalid method. Choose 'lsr' or 'pinv'.")

        # Compute projected regression value with TgtBar shift
        linreg_values[i] = intercept + slope * (length - 1 - tgt_bar)

    return pd.Series(linreg_values, index=price_series.index).bfill()


# =========================================
# Leavitt Projection & Convolution
# =========================================


def leavitt_projection(price_series, length, method="lsr"):
    """
    Compute the Leavitt Projection, a 1-bar forward forecast using Linear Regression.

    Parameters:
    - price_series (pd.Series): Input time series.
    - length (int): Lookback period for regression.
    - method (str, optional): Regression method ("lsr" or "pinv"). Default is "lsr".

    Returns:
    - pd.Series: Leavitt Projection values.
    """
    return ta_linreg(price_series, length, tgt_bar=-1, method=method)


def leavitt_convolution(price_series, plength, clength, method="lsr"):
    """
    Compute the Leavitt Convolution, which smooths the Leavitt Projection.

    Parameters:
    - price_series (pd.Series): Input time series.
    - plength (int): Lookback period for Leavitt Projection.
    - clength (int): Lookback period for Leavitt Convolution.
    - method (str, optional): Regression method ("lsr" or "pinv"). Default is "lsr".

    Returns:
    - tuple:
        - pd.Series: Leavitt Convolution values.
        - pd.Series: Slopes of the Convolution.
        - pd.Series: Intercepts of the Convolution.
    """
    # Compute Leavitt Projection
    leavitt_proj = leavitt_projection(price_series, plength, method=method)

    # Compute Leavitt Convolution using linear regression
    leavitt_conv = ta_linreg(leavitt_proj, clength, tgt_bar=-1, method=method)

    # Compute Slopes (Rate of Change)
    slopes = leavitt_conv.diff()  # First derivative

    # Compute Intercepts (Baseline at x=0)
    intercepts = leavitt_conv - slopes * (clength / 2)  # Adjusted for stability

    return leavitt_conv, slopes, intercepts


def leavitt_acceleration(lc_slope_series):
    """
    Compute the Leavitt Acceleration (LCACCELERATION), which measures
    the rate of change of the slope of the Leavitt Convolution.

    Parameters:
    - lc_slope_series (pd.Series): Leavitt Convolution Slope values.

    Returns:
    - pd.Series: Acceleration values.
    """
    return lc_slope_series.diff()


# =========================================
# Convolution Probability Function
# =========================================


def convolution_probability(lc_slope_series, lc_intercept_series, window=10):
    """
    Compute the Convolution Probability Function, which helps identify
    potential market turning points.

    Parameters:
    - lc_slope_series (pd.Series): Leavitt Convolution Slope values.
    - lc_intercept_series (pd.Series): Leavitt Convolution Intercept values.
    - window (int, optional): Rolling window size for standard deviation calculation. Default is 10.

    Returns:
    - pd.Series: Probability values indicating potential turning points.
    """
    slope_std = lc_slope_series.rolling(window=window).std()
    intercept_std = lc_intercept_series.rolling(window=window).std()

    # Compute Z-score probability metrics
    prob_slope = (lc_slope_series - lc_slope_series.mean()) / (slope_std + 1e-6)
    prob_intercept = (lc_intercept_series - lc_intercept_series.mean()) / (
        intercept_std + 1e-6
    )

    # Compute final probability estimate
    return prob_slope * (1 - prob_intercept)
