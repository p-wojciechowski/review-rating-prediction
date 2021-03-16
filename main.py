from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_rating
from starlette.responses import RedirectResponse

app = FastAPI(
    title = "Hotel Review Rating Predictor",
    description = 'ML prediction model served as REST API. Predicts how user rated hotel (from 1 to 5), based on his text review.'
)

class PredictPostBody(BaseModel):
    review_text: str

class ResponseModel(BaseModel):
    predicted_rating: float


@app.post('/predict', response_model=ResponseModel, status_code=200)
def predict(payload :PredictPostBody):
    text = payload.review_text
    predicted_rating = predict_rating(text)
    return {'predicted_rating' : predicted_rating}

@app.get("/status")
def check_status():
    return {'status':'App is working.'}

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url='/docs')