import torch
import os
from pymilvus import AnnSearchRequest, WeightedRanker

from image_search.extraction import FigureExtractor
from image_search.embedding import FigureVectorizer


IMAGES_DIR = 'static/images'
os.makedirs(IMAGES_DIR, exist_ok=True)


def get_image_path(id):
    return os.path.join(IMAGES_DIR, str(id) + '.jpg')


TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


def get_paper_path(arxiv_id) -> str:
    return os.path.join(TMP_DIR, arxiv_id + ".pdf")


def load_models():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)
    return extractor, vectorizer


def hybrid_search(client, image_embedding, caption_embedding, limit=10, img_weight=0.5):
    image_req = AnnSearchRequest(
        data=image_embedding, 
        anns_field="chart_embedding", 
        limit=limit, 
        param={'nprobe': 10}
    )

    caption_req = AnnSearchRequest(
        data=caption_embedding, 
        anns_field="caption_embedding", 
        limit=limit, 
        param={'nprobe': 10}
    )

    results = client.hybrid_search(
        collection_name="images_collection",
        reqs=[image_req, caption_req],
        ranker=WeightedRanker(img_weight, 1 - img_weight),
        limit=limit,
    )

    return results


from image_search.stream import PageStream
