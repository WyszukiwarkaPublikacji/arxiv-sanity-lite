from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer
from image_search.db import images_session, ImageModel
from aslite.db import get_embeddings_db
import sqlite3
import logging
import torch
import os
import requests
import io


def download_pdf(url):
    with requests.get(url) as r:
        r.raise_for_status()
        return io.BytesIO(r.content)


def extract_version(arxiv_id):
    """
    Extract the version suffix (e.g., 'v1', 'v2') from an arxiv_id.
    Returns 0 if no version suffix is present.
    """
    if 'v' in arxiv_id:
        version = arxiv_id.split('v')[-1]
        return int(version) if version.isdigit() else 0
    return 0


def get_non_matching_ids(papers_db_connection, images_session) -> list[str]:
    """
    Find all arxiv_ids in papers.db that are either missing in images.db
    or have a newer version available.
    """
    non_matching_ids = []
    seen_ids = set()

    cursor = papers_db_connection.cursor()
    cursor.execute("SELECT key FROM papers")
    papers_data = cursor.fetchall()

    for (arxiv_id,) in papers_data:
        if not arxiv_id:
            continue

        base_id = arxiv_id.split('v')[0]
        if base_id in seen_ids:
            continue

        seen_ids.add(base_id)
        matching_versions = images_session.query(ImageModel.arxiv_id).filter(ImageModel.arxiv_id.like(f"{base_id}%")).all()
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
    Delete records of older versions of the given arxiv_id in the images table.
    """
    base_id = arxiv_id.split('v')[0]

    old_versions = session.query(ImageModel).filter(
        ImageModel.arxiv_id.like(f"{base_id}v%")
    ).all()
    old_versions = [version for version in old_versions if extract_version(version.arxiv_id) < extract_version(arxiv_id)]

    for old_version in old_versions:
        session.delete(old_version)

    session.commit()


def save_to_database(session, arxiv_id, figure_path, paper_id):
    try:
        new_image = ImageModel(
            arxiv_id=arxiv_id,
            figure_path=figure_path,
            caption=None,  # No captions saved
            paper_id=paper_id  # Ensure a valid paper_id is passed
        )
        session.add(new_image)
        session.commit()
        logging.info(f"Saved {figure_path} to the database with paper_id: {paper_id}")
    except Exception as e:
        logging.error(f"Database error: {e}")
        session.rollback()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)

    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)

    papers_db_path = "data/papers.db"
    papers_db_connection = sqlite3.connect(papers_db_path)

    client = get_embeddings_db()

    update_ids = get_non_matching_ids(papers_db_connection, images_session)

    for arxiv_id in update_ids:
        delete_old_versions(images_session, arxiv_id)

        try:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf = download_pdf(pdf_url)

            figures, _ = zip(*extractor(pdf))

            extracted_dir = "./extracted"
            os.makedirs(extracted_dir, exist_ok=True)

            cursor = papers_db_connection.cursor()
            cursor.execute("SELECT key FROM papers WHERE key = ?", (arxiv_id,))
            result = cursor.fetchone()
            paper_id = result[0] if result else None

            if not paper_id:
                logging.error(f"Could not find paper_id for arxiv_id {arxiv_id}. Skipping...")
                continue

            for idx, figure in enumerate(figures):
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
                    paper_id=paper_id,  # Use real paper_id
                )

            embeddings = vectorizer(figures, [])
            # Generate integer IDs for Milvus
            data = [{"id": idx + 1, "embedding": emb} for idx, emb in enumerate(embeddings)]

            try:
                delete_ids = [d["id"] for d in data]  # Ensure IDs are integers
                client.delete("image_embeddings", delete_ids)
            except Exception as e:
                logging.error(f"Failed to delete embeddings for arxiv_id {arxiv_id}: {e}")

            client.delete("image_embeddings", [d["id"] for d in data])
            client.insert("image_embeddings", data)

            logging.info(f"Processed and embedded figures for arxiv_id: {arxiv_id}")

        except Exception as e:
            logging.error(f"An error occurred while processing {arxiv_id}: {e}")

    papers_db_connection.close()
