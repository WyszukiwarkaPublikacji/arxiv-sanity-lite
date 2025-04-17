import argparse
import asyncio
import io
import logging
import os
import aiohttp
from PIL import Image
import tqdm.asyncio

from aslite.db import get_embeddings_db, get_images_db, get_papers_db
from image_search import get_models, get_image_path

from queue import Queue
import threading
from tqdm.asyncio import tqdm


TMP_DIR = "tmp"


def get_paper_path(arxiv_id):
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, arxiv_id + ".pdf")


def split_id(aid) -> tuple[str, int]:
    aid = aid.split("v")
    version = int(aid[-1]) if aid[-1].isdigit() else 0
    return aid[0], version


def get_non_matching_ids(papers_db, images_db) -> list[str]:
    """Find all arxiv ids in papers.db that are either missing
    in data/images.db or have a newer version available."""
    non_matching_ids = set()

    for arxiv_id in papers_db:
        base_id, version = split_id(arxiv_id)
        matching_aid = False

        for info in images_db.values():
            if info["base_id"] == base_id:
                matching_aid = True

                if info["version"] < version:
                    non_matching_ids.add(arxiv_id)

        if not matching_aid:
            non_matching_ids.add(arxiv_id)

    return list(non_matching_ids)


def delete_id(edb, idb, arxiv_id) -> None:
    to_delete = []
    base_id, _ = split_id(arxiv_id)

    for id, info in idb.items():
        if info["base_id"] == base_id:
            to_delete.append(id)
            del idb[id]

    if to_delete:
        edb.delete("images_collection", to_delete)


async def fetch_paper(arxiv_id, q, session, sem):
    url = "https://arxiv.org/pdf/%s.pdf" % arxiv_id

    async with sem:
        async with session.get(url) as resp:
            resp.raise_for_status()
            path = get_paper_path(arxiv_id)
            
            total = int(resp.headers.get('content-length', 0))
            
            with open(path, "wb") as f, tqdm(total=total, unit='B', unit_scale=True, desc=arxiv_id) as bar:
                async for chunk in resp.content.iter_chunked(1024):
                    f.write(chunk)
                    bar.update(len(chunk))

            q.put(arxiv_id)
            logging.info("successfuly fetched %s" % arxiv_id)


def process_papers(q: Queue, idb, edb):
    extractor, vectorizer = get_models()
    last_id = max(map(int, idb.keys()), default=0) + 1

    while True:
        print(q.qsize())
        arxiv_id = q.get()

        if arxiv_id is None:
            q.task_done()
            break
        
        try:
            base_id, version = split_id(arxiv_id)
            path = get_paper_path(arxiv_id)

            figures, captions = zip(*extractor(path, verbose=False))

            new_ids = list(range(last_id, last_id + len(figures)))

            for id, figure, caption in zip(new_ids, figures, captions):
                fig_path = get_image_path(arxiv_id, id)
                Image.fromarray(figure).save(fig_path)

                idb[id] = dict(base_id=base_id, version=version, caption=caption)

            caption_emb, image_emb = vectorizer(captions, figures)

            data = [
                dict(id=id, image_embedding=x, caption_embedding=y)
                for id, x, y in zip(new_ids, image_emb, caption_emb)
            ]

            edb.insert("images_collection", data)

            nfig = len(figures)
            logging.info("Processed and embedded %d figures for %s" % (nfig, arxiv_id))

            last_id += nfig
            
        except Exception as e:
            logging.warning("Exception while processing %s: %s" % (arxiv_id, e))
        
        os.remove(path)
        q.task_done()


async def fetch_papers(arxiv_ids, q, max_concurrency):
    sem = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_paper(aid, q, session, sem) for aid in arxiv_ids]
        await asyncio.gather(*tasks)


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

    pdb = get_papers_db("r")
    idb = get_images_db("c")
    edb = get_embeddings_db()

    logging.info("obtaining non-matching ids...")
    non_matching_ids = get_non_matching_ids(pdb, idb)
    arxiv_ids = non_matching_ids[:args.num]
    logging.info("obtained non-matching ids from db of sizes %d, %d" % (len(pdb), len(idb)))

    q = Queue()
    
    for file in os.listdir(TMP_DIR):
        q.put(file.removesuffix('.pdf'))
    
    t = threading.Thread(target=process_papers, args=(q, idb, edb))
    t.start()

    asyncio.run(fetch_papers(arxiv_ids, q, max_concurrency=5))

    q.put(None)
    t.join()


if __name__ == "__main__":
    main()
