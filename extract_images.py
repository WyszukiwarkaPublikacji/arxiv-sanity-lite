from image_search.extract import FigureExtractor
from image_search.embedding import FigureVectorizer
from aslite.db import get_embeddings_db, get_papers_db, CompressedSqliteDict, PAPERS_DB_FILE
import logging
import torch
import os
import requests
import io
from PIL import Image


def get_images_db(flag="r", autocommit=True):
    assert flag in ["r", "c"]
    pdb = CompressedSqliteDict(
        PAPERS_DB_FILE, tablename="images", flag=flag, autocommit=autocommit
    )
    return pdb


def fetch_paper(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    with requests.get(url) as r:
        r.raise_for_status()
        return io.BytesIO(r.content)


def extract_base(arxiv_id) -> str:
    return arxiv_id.split("v")[0]


def extract_version(arxiv_id) -> int:
    """
    Extract the version suffix from an arxiv_id.
    Returns 0 if no version suffix is present.
    """
    if "v" in arxiv_id:
        version = arxiv_id.split("v")[-1]
        return int(version) if version.isdigit() else 0
    return 0


def get_non_matching_ids(papers_db, images_db) -> list[str]:
    """
    Find all arxiv_ids in papers.db that are either missing in images.db
    or have a newer version available.
    """
    non_matching_ids = set()

    for arxiv_id in papers_db:
        base_id = arxiv_id.split("v")[0]
        matching_arxiv_id = False

        for img_id, info in images_db.items():
            if info["base_id"] == base_id:
                matching_arxiv_id = True
                if info["version"] < extract_version(arxiv_id):
                    non_matching_ids.add(arxiv_id)

        if not matching_arxiv_id:
            non_matching_ids.add(arxiv_id)

    return list(non_matching_ids)


def get_image_ids(images_db, arxiv_ids) -> list[int]:
    ids = []
    base_ids = set(extract_base(id) for id in arxiv_ids)

    for img_id, info in images_db.items():
        if info["base_id"] in base_ids:
            ids.append(img_id)

    return ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)

    extracted_dir = "./extracted"
    os.makedirs(extracted_dir, exist_ok=True)

    extractor = FigureExtractor()
    vectorizer = FigureVectorizer(device)

    client = get_embeddings_db()
    papers_db = get_papers_db("r")
    images_db = get_images_db("c")
    
    last_idx = int(max(images_db.keys(), default=0))

    non_matching_ids = get_non_matching_ids(papers_db, images_db)
    delete_ids = get_image_ids(images_db, non_matching_ids)

    for img_id in delete_ids:
        del images_db[img_id]    
    
    for arxiv_id in non_matching_ids:
        base_id = extract_base(arxiv_id)
        version = extract_version(arxiv_id)

        pdf = fetch_paper(arxiv_id)
        figures, captions = zip(*extractor(pdf, verbose=False))

        for idx, figure in enumerate(figures):
            figure_path = f"{extracted_dir}/{arxiv_id}_figure{idx + 1}.png"
            try:
                Image.fromarray(figure).save(figure_path)
            except Exception as e:
                logging.error(f"Failed to save image for {arxiv_id}: {e}")
                continue

            images_db[last_idx + idx] = dict(
                base_id=base_id, version=version, figure_path=figure_path
            )

        embeddings = vectorizer(figures, captions)
        data = [
            {"id": last_idx + idx, "embedding": emb}
            for idx, emb in enumerate(embeddings)
        ]

        try:
            delete_ids = [d["id"] for d in data]
            client.delete("image_embeddings", delete_ids)
        except Exception as e:
            logging.error(f"Failed to delete embeddings for arxiv_id {arxiv_id}: {e}")

        client.insert("image_embeddings", data)

        logging.info(f"Processed and embedded figures for arxiv_id: {arxiv_id}")
