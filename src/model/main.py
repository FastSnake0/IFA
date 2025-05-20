from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from fastapi.responses import StreamingResponse, JSONResponse

from generator import generate_image

app = FastAPI()

class ModelInfo(BaseModel):
    name: str
    description: str

@app.get("/models", response_model=List[ModelInfo])
def get_models():
    return [{"name": "ip_adapter_dummy", "description": "Dummy image"}]


@app.post("/generate")
async def generate(payload: dict):
    try:
        image_bytes = await generate_image(payload)  # предполагается, что generate_image асинхронна
        return StreamingResponse(image_bytes, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})