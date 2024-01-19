from fastapi import FastAPI
from fastapi.logger import logger
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from .transformer import predict_freq


x
app = FastAPI(
    title="NLP Engineer Assignment",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")


# TODO: Add a route to the API that accepts a text input and uses the trained
# model to predict the number of occurrences of each letter in the text up to
# that point.

class InputText(BaseModel):
    text: str

class OutputText(BaseModel):
    prediction: str

@app.post("/count/", response_model=OutputText)
def count_freq(body: InputText):
    logger.info(f'text: {body}')
    pred = predict_freq("trained_model", body.text)
    output = {
        "prediction": pred
    }
    logger.info(f'prediction: {pred}')
    return output

@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")

