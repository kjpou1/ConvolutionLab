import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


from src.schemas.prediction_input_schema import PredictionInputSchema
from src.pipeline.predict_pipeline import PredictPipeline

from src.logger_manager import LoggerManager
from pydantic import ValidationError

logging = LoggerManager.get_logger(__name__)
application = Flask(__name__)

app = application

## Route for a home page


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            # Parse and validate the input data
            validated_data = PredictionInputSchema(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get(
                    "parental_level_of_education"
                ),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score")),
            )

            # Convert to DataFrame
            pred_df = validated_data.to_dataframe()
            logging.info(pred_df)

            # Perform prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info("Prediction successful.")

            return render_template("home.html", results=results[0])

        except ValidationError as e:
            logging.error(f"Validation Error: {e}")
            return render_template("home.html", error=f"Validation Error: {e.errors()}")

        except Exception as e:
            logging.error(f"Prediction Error: {e}")
            return render_template(
                "home.html", error="An error occurred during prediction."
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8097)
