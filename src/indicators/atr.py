import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    """
    Optimized ATR calculation that first computes True Range and then applies EMA for ATR.

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' prices.
        period (int): Number of periods for ATR calculation.

    Returns:
        pd.DataFrame: Updated DataFrame with 'ATR' column.
    """
    # Compute True Range first
    true_range = np.maximum.reduce(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ]
    )

    # Apply EMA to True Range to calculate ATR
    df["ATR"] = (
        pd.Series(true_range, index=df.index).ewm(span=period, adjust=False).mean()
    )

    return df
