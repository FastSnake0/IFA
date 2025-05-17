from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ModelInfo(BaseModel):
    name: str
    description: str

@app.get("/models", response_model=List[ModelInfo])
def get_models():
    return [{"name": "ip_adapter_dummy", "description": "Dummy image"}]

@app.post("/generate")
def generate_image(payload: dict):
    # Пока просто возвращает статичное изображение
    return FileResponse("static/fig1.png", media_type="image/jpeg")