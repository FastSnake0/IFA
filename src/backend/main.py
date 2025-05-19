from fastapi import FastAPI
import httpx
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from io import BytesIO

from database import Base, engine
import models
import uuid
from fastapi import Depends
from sqlalchemy.orm import Session
from database import SessionLocal
import os

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



Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/generate")
async def generate(req: GenRequest, db: Session = Depends(get_db)):
    # Отправляем запрос модели
    async with httpx.AsyncClient() as client:
        response = await client.post("http://model:8000/generate", json=req.dict())

    if response.status_code != 200:
        return {"error": "Error generating image"}

    # Сохраняем пользователя, если его нет
    user = db.query(models.User).filter_by(username=req.username).first()
    if not user:
        user = models.User(username=req.username)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Сохраняем изображение на диск
    image_bytes = response.content
    os.makedirs("images", exist_ok=True)
    filename = f"images/{uuid.uuid4().hex}.jpg"
    with open(filename, "wb") as f:
        f.write(image_bytes)

    # Сохраняем путь в БД
    db_image = models.Image(path=filename, owner=user)
    db.add(db_image)
    db.commit()

    # Возвращаем изображение
    return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")