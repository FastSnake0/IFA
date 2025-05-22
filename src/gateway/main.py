from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI()

class IPConfig(BaseModel):
    image: str  # base64 или URL
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
    ng_prompt: Optional[str]
    ip: Optional[dict]
    inject: Optional[dict]
    username: str

# Получение списка моделей
@app.get("/models")
async def get_models():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://backend:8000/models")
    response.raise_for_status()
    return response.json()

# Генерация изображения
@app.post("/generate")
async def generate(req: GenRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://backend:8000/generate", json=req.dict())
    if resp.status_code == 200:
        return Response(content=resp.content, media_type="image/jpeg")
    raise HTTPException(status_code=resp.status_code, detail=resp.text)

# Список пользователей
@app.get("/users")
async def list_users():
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://backend:8000/users")
    resp.raise_for_status()
    return resp.json()

# Информация о пользователе
@app.get("/users/{username}")
async def user_info(username: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://backend:8000/users/{username}")
    resp.raise_for_status()
    return resp.json()

# Удаление пользователя
@app.delete("/users/{username}")
async def delete_user(username: str):
    async with httpx.AsyncClient() as client:
        resp = await client.delete(f"http://backend:8000/users/{username}")
    if resp.status_code == 204:
        return Response(status_code=204)
    raise HTTPException(status_code=resp.status_code, detail=resp.text)

# Список изображений
@app.get("/images")
async def list_images(username: Optional[str] = None, date: Optional[str] = None):
    params = {}
    if username:
        params["username"] = username
    if date:
        params["date"] = date
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://backend:8000/images", params=params)
    resp.raise_for_status()
    return resp.json()

# Метаданные изображения
@app.get("/images/{image_id}")
async def image_info(image_id: int):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://backend:8000/images/{image_id}")
    resp.raise_for_status()
    return resp.json()

# Удаление изображения
@app.delete("/images/{image_id}")
async def delete_image(image_id: int):
    async with httpx.AsyncClient() as client:
        resp = await client.delete(f"http://backend:8000/images/{image_id}")
    if resp.status_code == 204:
        return Response(status_code=204)
    raise HTTPException(status_code=resp.status_code, detail=resp.text)
