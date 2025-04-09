"""Decimer test. This code will have to be moved somewhere else before merging to master"""

# https://arxiv.org/pdf/<id>

import requests
import pdf2image  # Debian: apt install poppler-utils / Alpine: apk add poppler
from PIL import Image
import numpy as np
import logging
import decimer_segmentation  # Debian: apt install libgl1
import DECIMER
from aslite.fingerprint import calculate_embedding
import rdkit.Chem
from aslite.db import EmbeddingsDB, get_papers_db
from aslite import config
from pymilvus import MilvusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_pdf_url(id_: str, source: str = "arxiv") -> str:
    if source == "arxiv":
        return f"https://arxiv.org/pdf/{id_}"
    elif source == "chemrxiv":
        r = requests.get(
            f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items/{id_}"
        )
        r.raise_for_status()
        return r.json()["asset"]["original"]["url"]
    else:
        raise ValueError("'source' must be equal to 'arxiv' or 'chemrxiv'.")


def download_pdf_as_images(url: str) -> list[np.ndarray]:
    # Warning: the whole paper is downloaded to RAM (TODO: check if it's really a problem at scale).
    r = requests.get(url)
    r.raise_for_status()
    images = pdf2image.convert_from_bytes(r.content)
    images = [np.array(img) for img in images]
    del r
    return images


def detect_chemical_structures(img: np.ndarray) -> list[np.ndarray]:
    return decimer_segmentation.segment_chemical_structures(
        img, expand=False, visualization=False
    )


def validate_smiles(smiles: str) -> bool:
    return rdkit.Chem.MolFromSmiles(smiles) is not None


def predict_smiles(
    img: np.ndarray, validate: bool = False, hand_drawn: bool = False
) -> tuple[str, float]:
    smiles, tokens = DECIMER.predict_SMILES(img, confidence=True, hand_drawn=hand_drawn)
    if validate and not validate_smiles(smiles):
        return smiles, 0.0
    confidence = np.mean(np.array([confidence for token, confidence in tokens]))
    return smiles, confidence

def process_paper(paper: dict, confidence_threshold: float = 0.0, embedding_db: MilvusClient | None = None) -> None:
    
    id_, source = paper["_id"], paper["provider"]
    print(id_)
    logging.info("Paper %s/%s: downloading" % (source, id_))
    url = get_pdf_url(id_=id_, source=source)
    imgs = download_pdf_as_images(url)

    detected = []
    for idx, img in enumerate(imgs):
        logging.info("Paper %s/%s: detecting structures on page %d/%d" % (source, id_, idx + 1, len(imgs)))
        detected += detect_chemical_structures(img)

    smiles = []
    confidence = []
    for idx, img in enumerate(detected):
        logging.info("Paper %s/%s: extracting SMILES from detected structure %d/%d" % (source, id_, idx + 1, len(detected)))

        smiles_i, confidence_i = predict_smiles(img)
        if confidence_i < confidence_threshold:
            continue

        emb = calculate_embedding(smiles_i)  # TODO: try ValueError: couldnt convert smiles to fingerprint
        embedding_db.insert(
            collection_name="chemical_embeddings",
            data=[
                {
                    "chemical_embedding": emb,
                    "tags": [j["term"] for j in paper["tags"]],
                    "category": "chemistry",
                    "paper_id": id_,
                    "SMILES": smiles_i,
                }
            ],
        )
        logging.info("Paper %s/%s: added embedding to the DB: %f %s" % (source, id_, confidence_i, smiles_i))

    logging.info("Paper %s/%s: finished processing." % (source, id_))


def start_processing():
    # TODO: run the processing for multiple papers in parallel
    logging.info("Started processing")

    pdb = get_papers_db(flag="c")
    with EmbeddingsDB() as embedding_db:
        for id_ in pdb:
            if pdb[id_]["provider"] != "chemrxiv":  # TODO: find a better criterion?
                continue
            try:
                process_paper(paper=pdb[id_], embedding_db=embedding_db)
            except Exception as err:
                logging.error("Paper %s/%s: processing failed: %s" % (pdb[id_]["provider"], id_, repr(err)))

if __name__ == "__main__":
    start_processing()
