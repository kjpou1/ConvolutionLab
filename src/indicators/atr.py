import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, window: int = 14):
    """
    Optimized ATR calculation that first computes True Range and then applies EMA for ATR.

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' prices.
        window (int): Number of periods (window) for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    # Compute True Range
    true_range = np.maximum.reduce(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ]
    )

    # Apply Exponential Moving Average (EMA) to True Range to calculate ATR
    atr = pd.Series(true_range, index=df.index).ewm(span=window, adjust=False).mean()
    return atr
