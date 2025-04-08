import os
from PIL import Image
from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from extract import FigureExtractor

# Logging setup
import logging

logging.basicConfig(level=logging.INFO)

# Database setup
Base = declarative_base()


class PublicationModel(Base):
    __tablename__ = "publications"
    paper_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)


class ImageModel(Base):
    __tablename__ = "figures"
    figure_id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(12), nullable=False)
    figure_path = Column(String, nullable=False)
    caption = Column(Integer, ForeignKey("captions.caption_id"), nullable=True)
    paper_id = Column(Integer, ForeignKey("publications.paper_id"), nullable=False)


class FigureModel(Base):
    __tablename__ = "captions"
    caption_id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(Text, nullable=True)


DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base.metadata.create_all(engine)


def save_to_database(arxiv_id, figure_path, description=None, paper_id=None):
    try:
        caption_id = None
        if description:
            caption = (
                session.query(FigureModel).filter_by(description=description).first()
            )
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
    arxiv_id = os.path.splitext(os.path.basename(pdf_path))[0]
    figures, captions = list(zip(*extractor(pdf_path, verbose=False)))

    extracted_dir = os.path.join(os.getcwd(), "extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    for idx, (figure, caption) in enumerate(zip(figures, captions)):
        image_name = f"{arxiv_id}_figure{idx + 1}.png"
        image_path = os.path.join(extracted_dir, image_name)
        Image.fromarray(figure).save(image_path)

        save_to_database(arxiv_id, image_path, caption, paper_id)
