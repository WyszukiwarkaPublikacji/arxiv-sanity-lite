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
    
    for arxiv_id in update_ids:  
        link = [l['href'] for l in papers_db[id]['links'] if l['title'] == 'pdf'][0]
        pdf = download_pdf(link)
        
        figures, captions = zip(*extractor(pdf))
        
        # insert images to db and obtain primary keys based on arxiv_id
        # question: are these ids ordered?
        image_ids = ...
        
        embeddings = vectorizer(figures, captions)
        data = [{"id": id, "embedding": emb} for id, emb in zip(image_ids, embeddings)]
        
        client.delete("image_embeddings", image_ids)
        client.insert("image_embeddings", data)
        