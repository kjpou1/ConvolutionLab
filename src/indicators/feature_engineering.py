import pandas as pd

import src.indicators.leavitt_indicator as lu
from src.indicators.atr import calculate_atr


def compute_indicators(df: pd.DataFrame, atr_period: int = 14):
    """Computes returns, momentum, and time-based features."""
    df["Returns"] = df["Close"].pct_change()
    lag_periods = [1, 2, 5, 10, 21]

    for lag in lag_periods:
        df[f"Returns_T-{lag}"] = df["Returns"].shift(lag)
        df[f"Momentum_T-{lag}"] = df["Returns"].sub(df["Returns"].shift(lag))

    df["Hour"] = df.index.hour
    df["Day_Of_Week"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Year"] = df.index.year

    # Ensure ATR is computed first
    df["ATR"] = calculate_atr(df, period=atr_period)

    return df


def compute_leavitt_data(df: pd.DataFrame):
    """Computes Leavitt-related indicators."""
    ahma_period, plength, clength = 9, 9, 3
    df["AHMA"] = lu.adaptive_hull_moving_average(df["Close"], ahma_period)
    df["Leavitt_Projection"] = lu.leavitt_projection(df["AHMA"], plength)
    df["Leavitt_Convolution"], df["LC_Slope"], df["LC_Intercept"] = (
        lu.leavitt_convolution(df["AHMA"], plength, clength)
    )
    df["LC_Acceleration"] = lu.leavitt_acceleration(df["LC_Slope"])
    df["Convolution_Probability"] = lu.convolution_probability(
        df["LC_Slope"], df["LC_Intercept"], window=10
    )

    return df
