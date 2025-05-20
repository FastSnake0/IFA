from fastapi import FastAPI, Response
import httpx
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel
from typing import Optional

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
    ip: Optional[str]
    inject: Optional[str]
    username: str

@app.get("/models")
async def get_models():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://backend:8000/models")
    return response.json()

@app.post("/generate")
async def generate(req: GenRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://backend:8000/generate", json=req.dict())
    if resp.status_code == 200:
        # Просто возвращаем bytes
        return Response(content=resp.content, media_type="image/jpeg")
    return {"error": "Error generating image"}