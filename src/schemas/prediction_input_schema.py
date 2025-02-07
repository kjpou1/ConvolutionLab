import pandas as pd
from pydantic import BaseModel, Field


class PredictionInputSchema(BaseModel):
    gender: str = Field(
        ...,
        pattern="^(male|female)$",
        description="Gender should be 'male' or 'female'.",
    )
    race_ethnicity: str = Field(
        ..., description="Race/Ethnicity must be a valid category."
    )
    parental_level_of_education: str = Field(
        ..., description="Parental level of education is required."
    )
    lunch: str = Field(
        ...,
        pattern="^(standard|free/reduced)$",
        description="Lunch must be 'standard' or 'free/reduced'.",
    )
    test_preparation_course: str = Field(
        ...,
        pattern="^(none|completed)$",
        description="Test preparation course must be 'none' or 'completed'.",
    )
    reading_score: float = Field(
        ..., ge=0, le=100, description="Reading score must be between 0 and 100."
    )
    writing_score: float = Field(
        ..., ge=0, le=100, description="Writing score must be between 0 and 100."
    )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the validated input to a pandas DataFrame.
        """
        return pd.DataFrame([self.model_dump()])
