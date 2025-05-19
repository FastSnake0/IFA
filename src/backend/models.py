from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)

    images = relationship("Image", back_populates="owner")

class Image(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String)
    user_id = Column(Integer, ForeignKey("user.id"))

    owner = relationship("User", back_populates="images")