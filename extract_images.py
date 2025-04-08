from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer
from aslite.db import get_embeddings_db, get_papers_db, get_metas_db

import logging
import torch
import requests
import io


def download_pdf(url):
    with requests.get(url) as r:
        r.raise_for_status()
        stream = io.BytesIO(r.content)
    return stream


def extract_version(arxiv_id):
    """
    Extract the version suffix (e.g., 'v1', 'v2') from an arxiv_id.
    Returns 0 if no version suffix is present.
    """
    if 'v' in arxiv_id:
        version = arxiv_id.split('v')[-1]
        if version.isdigit():
            return int(version)
    return 0


def get_non_matching_ids(papers_db, images_db) -> list[str]:
    """
    Retrieve a list of arxiv_ids not present in the images database.
    Accounts for new versions like v1, v2, etc.
    """
    non_matching_ids = []
    seen_ids = set()  # Track seen IDs to only process the latest version

    # Iterate through arxiv_ids in the papers database
    for paper_id, metadata in papers_db.items():
        arxiv_id = metadata.get('arxiv_id')
        print(arxiv_id)
        if not arxiv_id:
            continue

        base_id = arxiv_id.split('v')[0]
        if base_id in seen_ids:
            continue  # Skip if already processed this base ID

        seen_ids.add(base_id)
        matching_versions = [key for key in images_db.keys() if key.startswith(base_id)]

        # If no match is found, add the arxiv_id to non_matching_ids
        if not matching_versions:
            non_matching_ids.append(arxiv_id)
        else:
            # Identify the latest version in the figures database
            latest_version_in_db = max(matching_versions, key=extract_version)
            if extract_version(arxiv_id) > extract_version(latest_version_in_db):
                non_matching_ids.append(arxiv_id)

    return non_matching_ids


def delete_old_versions(images_db, arxiv_id):
    """
    Delete older versions of a publication along with connected records from the figures database.
    """
    base_id = arxiv_id.split('v')[0]  # Extract base ID without version
    versions_to_delete = [
        key for key in images_db.keys()
        if key.startswith(base_id) and extract_version(key) < extract_version(arxiv_id)
    ]

    # Delete records for older versions
    for version in versions_to_delete:
        associated_figures = images_db[version].get("figure_id", [])
        for figure_id in associated_figures:
            del images_db[figure_id]
        del images_db[version]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.basicConfig()
    
    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)

    papers_db = get_papers_db(flag='r')
    client = get_embeddings_db()
    images_db = get_metas_db(flag='c')

    update_ids = get_non_matching_ids(papers_db, images_db)

    # Process each paper with a non-matching ID
    for arxiv_id in update_ids:
        delete_old_versions(images_db, arxiv_id)

        link = [l['href'] for l in papers_db[id]['links'] if l['title'] == 'pdf'][0]
        pdf = download_pdf(link)

        figures, captions = zip(*extractor(pdf))

        # Insert figures into the database and get image IDs
        with get_metas_db(flag='c') as metas_db:
            image_ids = []
            for figure, caption in zip(figures, captions):
                figure_id = metas_db.insert(figure, caption, arxiv_id=arxiv_id)
                image_ids.append(figure_id)

        embeddings = vectorizer(figures, captions)
        data = [{"id": id, "embedding": emb} for id, emb in zip(image_ids, embeddings)]

        client.delete("image_embeddings", image_ids)
        client.insert("image_embeddings", data)
