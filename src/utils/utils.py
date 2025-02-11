# Helper function to safely convert to float
def safe_float(value):
    return (
        float(value) if value is not None else float("nan")
    )  # Or use 0.0 as default if needed
