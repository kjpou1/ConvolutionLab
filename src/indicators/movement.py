import numpy as np


def classify_movement(df, period=14, scale_factor=0.25):
    """
    Classifies price movement based on ATR-derived volatility thresholds.

    Args:
        df (pd.DataFrame): Market data.
        period (int): ATR period.
        scale_factor (float): Multiplier for ATR dynamic threshold.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Movement_Class'.
    """

    # Compute dynamic ATR multiplier
    df["atr_std"] = df["ATR"].rolling(window=period).std()
    df["dynamic_atr_multiplier"] = (
        scale_factor + (df["atr_std"] / df["ATR"].mean()) * 0.5
    )

    # Compute movement classification
    df["price_change"] = df["Close"].diff()
    df["dynamic_volatility_threshold"] = df["ATR"] * df["dynamic_atr_multiplier"]

    df["Movement_Class"] = np.select(
        [
            df["price_change"] > df["dynamic_volatility_threshold"],
            df["price_change"] < -df["dynamic_volatility_threshold"],
        ],
        [2, 0],  # Up, Down
        default=1,  # Neutral
    )

    # Cleanup
    df.drop(
        columns=[
            "price_change",
            "dynamic_volatility_threshold",
            "atr_std",
            "dynamic_atr_multiplier",
        ],
        inplace=True,
    )

    return df
