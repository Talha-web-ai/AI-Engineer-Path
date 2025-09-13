from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    feature1: float = Field(..., description="First numeric feature")
    feature2: float = Field(..., description="Second numeric feature")
    feature3: float = Field(..., description="Third numeric feature")

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
