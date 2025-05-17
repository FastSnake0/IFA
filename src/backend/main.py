from fastapi import FastAPI
import httpx
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from io import BytesIO

app = FastAPI()

class IPConfig(BaseModel):
    image: str  # base64 или url
    scale: float
    start_at: float
    end_at: float

class InjectConfig(BaseModel):
    layer: str
    scale: float
    start_at: float
    end_at: float

class GenRequest(BaseModel):
    model: str
    pos_prompt: str
    ng_prompt: str
    ip: Optional[IPConfig]
    inject: Optional[InjectConfig]

@app.post("/generate")
async def generate(req: GenRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://model:8000/generate", json=req.dict())

    # Если ответ от модели — это изображение, обрабатываем его как бинарные данные
    if response.status_code == 200:
        image_data = response.content  # бинарные данные изображения
        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")  # или другой тип изображения
    else:
        return {"error": "Error generating image"}