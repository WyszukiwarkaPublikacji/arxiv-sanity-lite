import torch
import os

from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer


IMAGES_DIR = 'static/images'
os.makedirs(IMAGES_DIR, exist_ok=True)


def get_models():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)
    return extractor, vectorizer


def get_image_path(aid, id):
    return os.path.join(IMAGES_DIR, f'{aid}_{id}.png')