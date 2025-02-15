import argparse
import os
import signal
import sys

import joblib  # For loading the trained model
import pandas as pd

from src.logger_manager import LoggerManager
from src.pipeline.predict_pipeline import PredictPipeline  # Ensure src is accessible

logging = LoggerManager.get_logger(__name__)


def handle_interrupt(signal, frame):
    logging.warning("Process interrupted. Shutting down gracefully...")
    sys.exit(1)


signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)


# Validate OHLCV format
def validate_ohlcv(record):
    """
    Validates whether the input record follows the required OHLCV format.
    Ensures correct data structure, types, and valid `Date` formats.
    """
    expected_keys = ["Date", "Open", "High", "Low", "Close", "Volume"]

    if isinstance(record, dict):
        if not all(key in record for key in expected_keys):
            raise ValueError(
                f"Invalid OHLCV record: Missing required keys. Expected: {expected_keys}. Received: {list(record.keys())}"
            )
        try:
            pd.to_datetime(record["Date"])
        except Exception:
            raise ValueError(f"Invalid `Date` format: {record['Date']}")
        if not all(isinstance(record[key], (int, float)) for key in expected_keys[1:]):
            raise ValueError(
                f"Invalid OHLCV record: Values must be int/float for OHLCV fields. Received: {record}"
            )
    elif isinstance(record, list):
        if len(record) != 6:
            raise ValueError(
                f"Invalid OHLCV record: Expected 6 elements (Date + OHLCV), got {len(record)}"
            )
        try:
            pd.to_datetime(record[0])
        except Exception:
            raise ValueError(f"Invalid `Date` format: {record[0]}")
        if not all(isinstance(value, (int, float)) for value in record[1:]):
            raise ValueError(
                f"Invalid OHLCV record: Values must be int/float for OHLCV fields. Received: {record}"
            )
    else:
        raise TypeError("Invalid input type. Record must be a dictionary or list.")
    return True


# Validate batch of OHLCV records
def validate_ohlcv_batch(records):
    """
    Validates a batch of OHLCV records using assertions.
    """
    results = []
    for i, record in enumerate(records):
        try:
            assert validate_ohlcv(record) == True
            results.append(True)
        except ValueError as e:
            logging.error(f"AssertionError: Record {i} failed validation: {e}")
            results.append(False)
    return results


# Reformat dataset to DOHLCV format
def reformat_to_dohlcv(df: pd.DataFrame):
    """
    Converts a dataset from mid_* format to DOHLCV format.
    """
    try:
        if "time" in df.columns:
            df = df.rename(columns={"time": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
        columns_to_drop = [
            "bid_o",
            "bid_h",
            "bid_l",
            "bid_c",
            "ask_o",
            "ask_h",
            "ask_l",
            "ask_c",
        ]
        df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
            inplace=True,
        )
        rename_mapping = {
            "mid_o": "Open",
            "mid_h": "High",
            "mid_l": "Low",
            "mid_c": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_mapping)
        dohlcv_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[dohlcv_columns]
        logging.info("Dataset successfully reformatted to DOHLCV format.")
        return df
    except Exception as e:
        logging.error(f"Error during reformatting: {e}")
        return None


# Load data
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    logging.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df = reformat_to_dohlcv(df)
    df_validated = df.to_dict(orient="records")
    validate_ohlcv_batch(df_validated)
    return df


# Run predictions
def run_prediction(df):
    logging.info("Initializing prediction pipeline...")
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(df)

    prediction_count = len(result.predictions)
    total_records = len(df)

    if prediction_count != total_records:
        logging.warning(
            f"Mismatch: Input records ({total_records}) and predictions ({prediction_count}). Adjusting output..."
        )

    df_subset = df.tail(prediction_count).reset_index(drop=True)
    df_predictions = pd.DataFrame(result._asdict())
    df_final = pd.concat([df_subset, df_predictions], axis=1)

    return df_final


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OHLCV prediction pipeline.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input OHLCV CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV file (default: predictions.csv).",
    )
    args = parser.parse_args()

    try:
        df_ohlcv = load_data(args.input)
        df_predictions = run_prediction(df_ohlcv)
        df_predictions.to_csv(args.output, index=False)
        logging.info(f"Predictions saved to {args.output}")
    except Exception as e:
        logging.error(f"Error: {e}")
