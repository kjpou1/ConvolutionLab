from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    code: int
    code_text: str
    message: str
    data: Optional[Dict[str, float]] = None
