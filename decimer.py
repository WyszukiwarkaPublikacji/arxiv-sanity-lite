import logging

import DECIMER
import decimer_segmentation
import numpy as np
import pdf2image
import rdkit.Chem
import requests
from PIL import Image
from pymilvus import MilvusClient

from aslite import config
from aslite.db import EmbeddingsDB, get_papers_db
from aslite.fingerprint import calculate_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_pdf_url(id_: str, source: str = "arxiv") -> str:
    if source == "arxiv":
        return f"https://arxiv.org/pdf/{id_}"
    elif source == "chemrxiv":
        r = requests.get(f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items/{id_}")
        r.raise_for_status()
        return r.json()["asset"]["original"]["url"]
    else:
        raise ValueError("'source' must be equal to 'arxiv' or 'chemrxiv'.")


def download_pdf_as_images(url: str) -> list[np.ndarray]:
    r = requests.get(url)
    r.raise_for_status()
    images = pdf2image.convert_from_bytes(r.content)
    images = [np.array(img) for img in images]
    return images


def detect_chemical_structures(img: np.ndarray) -> list[np.ndarray]:
    return decimer_segmentation.segment_chemical_structures(img, expand=False, visualization=False)


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


def process_paper(
    paper: dict,
    confidence_threshold: float = 0.0,
    embedding_db: MilvusClient | None = None,
) -> None:

    id_, source = paper["_id"], paper["provider"]
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
        logging.info("Paper %s/%s: SMILES predicted: %f %s" % (source, id_, confidence_i, smiles_i))
        if confidence_i < confidence_threshold:
            continue

        emb = calculate_embedding(smiles_i)
        if emb is None:
            logging.info("Paper %s/%s: SMILES embedding calculation failed: %s" % (source, id_, smiles_i))
            continue

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
        logging.info("Paper %s/%s: added SMILES embedding to the DB: %s" % (source, id_, smiles_i))

    logging.info("Paper %s/%s: processing finished." % (source, id_))


def start_processing():
    # TODO: run the processing for multiple papers in parallel
    logging.info("Started processing")

    pdb = get_papers_db(flag="c")
    with EmbeddingsDB() as embedding_db:
        for id_ in pdb:
            if pdb[id_]["provider"] != "chemrxiv":  # TODO: find a better criterion?
                continue
            print(pdb[id_], pdb[id_]["computed_chemical"])
            if pdb[id_]["computed_chemical"]:
                continue
            try:
                process_paper(paper=pdb[id_], embedding_db=embedding_db)
            except Exception as err:
                logging.error("Paper %s/%s: processing failed: %s" % (pdb[id_]["provider"], id_, repr(err)))
            else:
                el = pdb[id_]
                el["computed_chemical"] = True
                pdb[id_] = el


if __name__ == "__main__":
    start_processing()
