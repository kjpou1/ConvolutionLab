from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.models.prediction_request import PredictionRequest
from src.models.prediction_response import PredictionResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.schemas.prediction_input_schema import PredictionInputSchema
from src.logger_manager import LoggerManager

# Initialize the FastAPI application
app = FastAPI()

# Initialize the logger
logging = LoggerManager.get_logger(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()


@app.get("/")
def read_root():
    """
    Root endpoint to verify that the API is running.
    """
    return {"message": "FastAPI Prediction Service is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    """
    Endpoint to handle prediction requests.

    Args:
        data (PredictionRequest): Input data for prediction.

    Returns:
        PredictionResponse: Result with status and math score.
    """
    try:
        # Extract and validate the input data
        inner_data = data.payload.get("data", {})
        validated_data = PredictionInputSchema(**inner_data)
        input_data = validated_data.to_dataframe()

        logging.info("Input data validated and converted to DataFrame.")

        # Perform prediction
        prediction = predict_pipeline.predict(input_data)
        logging.info("Prediction successful.")

        # Return successful response with 200 OK
        return PredictionResponse(
            code=0,
            code_text="ok",
            message="Processed successfully.",
            data={"math_score": prediction[0]},
        )

    except ValidationError as e:
        # Parse validation errors
        error_details = e.errors()
        formatted_errors = [
            {"field": ".".join(map(str, err["loc"])), "error": err["msg"]}
            for err in error_details
        ]
        logging.error(f"Validation Error: {formatted_errors}")

        # Return 400 Bad Request with detailed error response
        raise HTTPException(
            status_code=400,
            detail={
                "code": -1,
                "code_text": "error",
                "message": "Validation error occurred.",
                "errors": formatted_errors,
            },
        )

    except Exception as e:
        # Log unexpected errors
        logging.error(f"Unexpected Error: {str(e)}")

        # Return 500 Internal Server Error with detailed error response
        raise HTTPException(
            status_code=500,
            detail={
                "code": -1,
                "code_text": "error",
                "message": "An internal server error occurred.",
                "errors": None,
            },
        )


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8008)
