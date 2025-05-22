from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import httpx
import os
import uuid
from io import BytesIO

from database import Base, engine, SessionLocal
import models

app = FastAPI()

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class IPConfig(BaseModel):
    image: str
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

@app.post("/generate")
async def generate(req: GenRequest, db: Session = Depends(get_db)):
    # Обработка запроса к модели
    async with httpx.AsyncClient() as client:
        response = await client.post("http://model:8000/generate", json=req.dict())
    
    # Проверка ответа от модели
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error generating image")
    
    # Проверка, если пользователь существует, и его создание
    user = db.query(models.User).filter_by(username=req.username).first()
    if not user:
        user = models.User(username=req.username)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Сохранение изображения на диск с уникальным именем
    os.makedirs("images", exist_ok=True)
    filename = f"images/{uuid.uuid4().hex}.jpg"
    try:
        with open(filename, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image: {e}")
    
    # Добавление записи в базу данных
    db_image = models.Image(path=filename, owner_id=user.id)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    # Возвращаем изображение как StreamingResponse
    return StreamingResponse(BytesIO(response.content), media_type="image/jpeg")

@app.get("/models")
async def list_models():
    return {
        "models": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0"
        ]
    }

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return [{"id": u.id, "username": u.username} for u in users]

@app.get("/users/{username}")
def user_info(username: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "username": user.username,
        "images": [{"id": img.id, "path": img.path} for img in user.images]
    }

@app.delete("/users/{username}")
def delete_user(username: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"detail": f"User {username} deleted"}

@app.get("/images")
def list_images(username: Optional[str] = None, date: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Image)
    if username:
        user = db.query(models.User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        query = query.filter_by(owner_id=user.id)
    images = query.all()
    return [{"id": img.id, "path": img.path, "owner_id": img.owner_id} for img in images]

@app.get("/images/{image_id}")
def image_info(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return {
        "id": image.id,
        "path": image.path,
        "owner_id": image.owner_id
    }

@app.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Удаляем файл с диска, если он существует
    if os.path.exists(image.path):
        try:
            os.remove(image.path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting image file: {e}")
    
    db.delete(image)
    db.commit()
    return {"detail": f"Image {image_id} deleted"}
