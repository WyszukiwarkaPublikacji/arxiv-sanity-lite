from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer
from aslite.db import get_embeddings_db, get_papers_db

import requests
import io


def download_pdf(url):
    with requests.get(url) as r:
        r.raise_for_status()
        stream = io.BytesIO(r.content)
    return stream


def get_non_matching_ids(papers_db, images_db) -> list[str]:
    """return ids that does not exist in
    images_db or its version doesn't match"""
    pass


if __name__ == '__main__':
    extractor = FigureExtractor()
    vectorizer = FigureVectorizer()
    
    papers_db = get_papers_db()
    client = get_embeddings_db()

    update_ids = get_non_matching_ids(papers_db, ...)
    
    # extract the figures from the records you will be adding
    # and insert them into the images_db and milvus collection.
    
    for arxiv_id in update_ids:
        # obtain primary key from images_db based on arxiv_id
        image_id = ...
        
        link = [l['href'] for l in papers_db[id]['links'] if l['title'] == 'pdf'][0]
        pdf = download_pdf(link)
        
        figures, captions = zip(*extractor(pdf))
        
        embeddings = vectorizer(figures, captions)
        data = [{"id": image_id, "embedding": emb} for emb in embeddings]
        
        client.insert("image_embeddings", data)
        