from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer
from aslite.db import get_embeddings_db, get_papers_db
from image_search.db import images_session, ImageModel, save_to_database

import logging
import torch
import requests
import io
import os


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


def get_non_matching_ids(papers_db, session) -> list[str]:
    """
    Find all arxiv_ids in papers.db that are either missing in images.db
    or have a newer version available.
    """
    non_matching_ids = []
    seen_ids = set()

    for paper in papers_db.values():
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            continue

        base_id = arxiv_id.split('v')[0]
        if base_id in seen_ids:
            continue

        seen_ids.add(base_id)
        # Query figures table for matching arxiv_id versions
        matching_versions = session.query(ImageModel.arxiv_id).filter(ImageModel.arxiv_id.like(f"{base_id}%")).all()
        matching_versions = [version[0] for version in matching_versions]

        if not matching_versions:
            non_matching_ids.append(arxiv_id)
        else:
            latest_version_in_db = max(matching_versions, key=extract_version)
            if extract_version(arxiv_id) > extract_version(latest_version_in_db):
                non_matching_ids.append(arxiv_id)

    return non_matching_ids


def delete_old_versions(session, arxiv_id):
    """
    Delete records of older versions of the given arxiv_id in the figures table.
    """
    base_id = arxiv_id.split('v')[0]

    # Get all matching versions from DB
    all_versions = session.query(ImageModel).filter(
        ImageModel.arxiv_id.like(f"{base_id}v%")
    ).all()

    # Only keep those with lower version number
    old_versions = [
        version for version in all_versions
        if extract_version(version.arxiv_id) < extract_version(arxiv_id)
    ]

    for old_version in old_versions:
        session.delete(old_version)

    session.commit()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO)

    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)

    papers_db = get_papers_db()
    client = get_embeddings_db()

    update_ids = get_non_matching_ids(papers_db, images_session)

    # Process each paper with a non-matching ID
    for arxiv_id in update_ids:
        delete_old_versions(images_session, arxiv_id)

        try:
            # Get the PDF download link from papers_db
            pdf_url = next(
                link['href'] for link in papers_db[arxiv_id]['links'] if link['title'] == 'pdf'
            )
            pdf = download_pdf(pdf_url)

            figures, captions = zip(*extractor(pdf))

            extracted_dir = "./extracted"
            os.makedirs(extracted_dir, exist_ok=True)

            paper_record = papers_db.get(arxiv_id)
            if not paper_record:
                logging.error(f"Paper record not found for arxiv_id: {arxiv_id}. Skipping...")
                continue

            paper_id = arxiv_id

            for idx, (figure, caption) in enumerate(zip(figures, captions)):
                figure_path = f"{extracted_dir}/{arxiv_id}_figure{idx + 1}.png"

                try:
                    from PIL import Image
                    Image.fromarray(figure).save(figure_path)
                except Exception as e:
                    logging.error(f"Failed to save image for {arxiv_id}: {e}")
                    continue

                save_to_database(
                    session=images_session,
                    arxiv_id=arxiv_id,
                    figure_path=figure_path,
                    description=caption,
                    paper_id=paper_id
                )

            embeddings = vectorizer(figures, captions)
            data = [{"id": f"{arxiv_id}_figure{idx + 1}", "embedding": emb} for idx, emb in enumerate(embeddings)]

            client.delete("image_embeddings", [d["id"] for d in data])
            client.insert("image_embeddings", data)

            logging.info(f"Processed and embedded figures for arxiv_id: {arxiv_id}")

        except KeyError:
            logging.error(f"No PDF link found for {arxiv_id}. Skipping...")
        except Exception as e:
            logging.error(f"An error occurred while processing {arxiv_id}: {e}")
