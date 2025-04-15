import argparse
import io
import logging
import os

import requests
import torch
from PIL import Image

from aslite.db import get_embeddings_db, get_images_db, get_papers_db
from image_search.embedding import FigureVectorizer
from image_search.extract import FigureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("extracted", exist_ok=True)

PAPER_URL_TEMPL = "https://arxiv.org/pdf/%s.pdf"


def fetch_paper(aid):
    url = PAPER_URL_TEMPL % aid

    with requests.get(url) as r:
        r.raise_for_status()
        return io.BytesIO(r.content)


def split_id(aid) -> tuple[str, int]:
    aid = aid.split("v")
    version = int(aid[-1]) if aid[-1].isdigit() else 0
    return aid[0], version


def get_non_matching_ids(papers_db, images_db) -> list[str]:
    """Find all arxiv ids in papers.db that are either
    missing in images.db or have a newer version available."""
    non_matching_ids = set()

    for aid in papers_db:
        base_id, _ = split_id(aid)
        matching_aid = False

        for info in images_db.values():
            if info["base_id"] == base_id:
                matching_aid = True
                _, version = split_id(aid)

                if info["version"] < version:
                    non_matching_ids.add(aid)

        if not matching_aid:
            non_matching_ids.add(aid)

    return list(non_matching_ids)


def delete_id(client, idb, aid) -> None:
    to_delete = []
    base_id, _ = split_id(aid)

    for id, info in idb.items():
        if info["base_id"] == base_id:
            to_delete.append(id)
            del idb[id]

    if to_delete:
        client.delete("images_collection", to_delete)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    parser = argparse.ArgumentParser(description="Extract images")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        help="up to how many papers to extract from",
    )
    args = parser.parse_args()
    print(args)

    extr = FigureExtractor()
    vect = FigureVectorizer(DEVICE)

    client = get_embeddings_db()
    pdb = get_papers_db("r")
    idb = get_images_db("c")

    last_id = int(max(idb.keys(), default=0))

    non_matching_ids = get_non_matching_ids(pdb, idb)

    for n, aid in enumerate(non_matching_ids):
        if n > args.num:
            break

        delete_id(client, idb, aid)

        try:
            base_id, version = split_id(aid)

            pdf = fetch_paper(aid)
            figures, captions = zip(*extr(pdf, verbose=False))

            new_ids = list(range(last_id, last_id + len(figures)))

            for id, figure in zip(new_ids, figures):
                fig_path = f"extracted/{aid}_{id}.png"
                Image.fromarray(figure).save(fig_path)

                idb[id] = dict(base_id=base_id, version=version)

            img_emb, cap_emb = vect(figures, captions)

            data = [
                {"id": id, "image_embedding": ie, "caption_embedding": ce}
                for id, ie, ce in zip(new_ids, img_emb, cap_emb)
            ]

            client.insert("images_collection", data)

            last_id += len(figures)
            logging.info("processed and embedded figures for %s" % aid)

        except Exception as e:
            logging.warning(e)


if __name__ == "__main__":
    import pyinstrument

    with pyinstrument.Profiler() as profiler:
        main()

    profiler.print()
