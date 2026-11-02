

from pydantic import BaseModel


class SMSRequest(BaseModel):
    message: str



class PredictionResponse(BaseModel):
    message: str
    prediction: str
    is_spam: bool
