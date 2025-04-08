import os
from PIL import Image
from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO)

# Database setup
Base = declarative_base()

class PublicationModel(Base):
    __tablename__ = "papers"
    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)

class ImageModel(Base):
    __tablename__ = "figures"
    figure_id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(12), nullable=False)
    figure_path = Column(String, nullable=False)
    caption = Column(Integer, ForeignKey("captions.caption_id"), nullable=True)
    paper_id = Column(String, ForeignKey("papers.key"), nullable=False)

class FigureModel(Base):
    __tablename__ = "captions"
    caption_id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text, nullable=True)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

papers_db_url = f"sqlite:///{os.path.join(data_dir, 'papers.db')}"
images_db_url = f"sqlite:///{os.path.join(data_dir, 'images.db')}"

papers_engine = create_engine(papers_db_url)
images_engine = create_engine(images_db_url)

def ensure_tables(engine, models):
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    for model in models:
        if model.__tablename__ not in existing_tables:
            logging.info(f"Creating table {model.__tablename__}")
            Base.metadata.create_all(engine, tables=[model.__table__])

# Ensure databases and tables
ensure_tables(papers_engine, [PublicationModel])
ensure_tables(images_engine, [ImageModel, FigureModel])

PapersSession = sessionmaker(bind=papers_engine)
ImagesSession = sessionmaker(bind=images_engine)
papers_session = PapersSession()
images_session = ImagesSession()

# Export sessions and models for external usage
__all__ = [
    'papers_session', 'images_session', 'save_to_database',
    'ImageModel', 'FigureModel', 'PublicationModel'
]

logging.info(
    "Loaded db.py. Available sessions: 'papers_session' for papers.db, 'images_session' for images.db."
)

def save_to_database(session, arxiv_id, figure_path, description=None, paper_id=None):
    """
    Save image data to the `figures` table in `images.db`.
    """
    try:
        caption_id = None
        if description:
            caption = session.query(FigureModel).filter_by(description=description).first()
            if not caption:
                caption = FigureModel(description=description)
                session.add(caption)
                session.commit()
            caption_id = caption.caption_id

        new_image = ImageModel(
            arxiv_id=arxiv_id,
            figure_path=figure_path,
            caption=caption_id,
            paper_id=paper_id,
        )
        session.add(new_image)
        session.commit()
        logging.info(f"Saved {figure_path} to the database")
    except Exception as e:
        logging.error(f"Database error: {e}")
        session.rollback()

def extract_and_store(extractor, pdf_path, paper_id):
    """
    Extract figures from a PDF and save them to disk and the database.
    """
    arxiv_id = os.path.splitext(os.path.basename(pdf_path))[0]
    figures, captions = list(zip(*extractor(pdf_path, verbose=False)))

    extracted_dir = os.path.join(os.getcwd(), "extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    for idx, (figure, caption) in enumerate(zip(figures, captions)):
        image_name = f"{arxiv_id}_figure{idx + 1}.png"
        image_path = os.path.join(extracted_dir, image_name)

        # Save the figure to disk
        try:
            Image.fromarray(figure).save(image_path)
        except Exception as e:
            logging.error(f"Failed to save image for {arxiv_id}: {e}")
            continue

        save_to_database(images_session, arxiv_id, image_path, caption, paper_id)
