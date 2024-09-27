#Database Setup

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Index
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_file = Column(String, nullable=False)
    page_number = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)   

class FAQ(Base):
    __tablename__ = 'faqs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=False)

class ImageEntry(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_file = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String, nullable=False)  
    caption = Column(Text, nullable=True)    #doenst work perfect yet

# Create indexes for faster retrieval
Index('idx_pdf_file', Chunk.pdf_file)
Index('idx_page_number', Chunk.page_number)
Index('idx_question', FAQ.question)
Index('idx_image_pdf_file', ImageEntry.pdf_file)
Index('idx_image_page_number', ImageEntry.page_number)
Index('idx_image_caption', ImageEntry.caption)

def init_db(db_path="sqlite:///manuals.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
