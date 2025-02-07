# Define the input schema for the prediction API
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    payload: dict
