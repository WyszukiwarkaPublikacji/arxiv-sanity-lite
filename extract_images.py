import argparse
import asyncio
import io
import logging
import os
import aiohttp
from PIL import Image

import torch
from aslite.db import get_embeddings_db, get_images_db, get_papers_db
from image_search.embedding import FigureVectorizer
from image_search.extract import FigureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("static/extracted", exist_ok=True)

PAPER_URL_TEMPL = "https://arxiv.org/pdf/%s.pdf"


async def fetch_paper_async(aid, session):
    url = PAPER_URL_TEMPL % aid
    async with session.get(url) as response:
        response.raise_for_status()
        return io.BytesIO(await response.read())


async def process_aid(aid, extr, vect, client, idb, last_id, session):
    try:
        base_id, version = split_id(aid)
        pdf = await fetch_paper_async(aid, session)
        figures, captions = zip(*extr(pdf, verbose=False))

        new_ids = list(range(last_id, last_id + len(figures)))

        for id, figure, caption in zip(new_ids, figures, captions):
            fig_path = f"static/extracted/{aid}_{id}.png"
            Image.fromarray(figure).save(fig_path)

            idb[id] = dict(base_id=base_id, version=version, caption=caption)

        img_emb, cap_emb = vect(captions, figures)

        data = [
            {"id": id, "image_embedding": e1, "caption_embedding": e2}
            for id, e1, e2 in zip(new_ids, img_emb, cap_emb)
        ]

        client.insert("images_collection", data)

        logging.info("Processed and embedded figures for %s" % aid)

        return len(figures)
    except Exception as e:
        logging.warning(f"Error processing {aid}: {e}")
        return 0


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


async def process_all(aids, extr, vect, client, idb, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)
    last_id = int(max(idb.keys(), default=0))

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(process_with_semaphore(semaphore, aid, extr, vect, client, idb, last_id, session))
            for aid in aids
        ]
        results = await asyncio.gather(*tasks)
    return sum(results)  # Total number of figures processed


async def process_with_semaphore(semaphore, aid, extr, vect, client, idb, last_id, session):
    async with semaphore:
        return await process_aid(aid, extr, vect, client, idb, last_id, session)


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
        help="Up to how many papers to extract from",
    )
    args = parser.parse_args()

    extr = FigureExtractor()
    vect = FigureVectorizer(DEVICE)

    client = get_embeddings_db()
    pdb = get_papers_db("r")
    idb = get_images_db("c")

    non_matching_ids = get_non_matching_ids(pdb, idb)
    aids_to_process = non_matching_ids[:args.num]

    import pyinstrument
    with pyinstrument.Profiler() as profiler:
        total_figures = asyncio.run(process_all(aids_to_process, extr, vect, client, idb, max_concurrency=50))

    logging.info(f"Total figures processed: {total_figures}")
    profiler.print()


if __name__ == "__main__":
    main()
