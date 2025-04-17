import argparse
import asyncio
import logging
import os
import aiohttp
from PIL import Image

from aslite.db import get_embeddings_db, get_images_db, get_papers_db
from image_search import get_models, get_image_path

from queue import Queue
import threading
from tqdm.asyncio import tqdm


TMP_DIR = "tmp"

def get_paper_path(arxiv_id) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, arxiv_id + ".pdf")


def _arxiv_id(base_id, version) -> str:
    if version > 0:
        base_id += f'v{version}'
    return base_id


def split_id(aid) -> tuple[str, int]:
    aid = aid.split("v")
    version = int(aid[-1]) if aid[-1].isdigit() else 0
    return aid[0], version


def get_non_matching_ids(pdb, idb) -> list[str]:
    """Find all arxiv ids in papers.db that are either missing
    in data/images.db or have a newer version available."""
    non_matching_ids = set()
    base_ids = {k: v for k, v in map(split_id, pdb.keys())}

    for d in idb.values():
        bid = d["base_id"]
        if bid in base_ids:
            v = d["version"]
            arxiv_id = _arxiv_id(bid, v)
            
            if v < base_ids[bid]:
                non_matching_ids.add(arxiv_id)
                
            base_ids.pop(arxiv_id)

    non_matching_ids.update(base_ids.keys())
    return list(non_matching_ids)


def delete_ids(edb, idb, arxiv_ids) -> None:
    to_delete = []
    base_ids = set(split_id(aid)[0] for aid in arxiv_ids)

    for id, info in idb.items():
        if info["base_id"] in base_ids:
            to_delete.append(id)
            del idb[id]

    if to_delete:
        edb.delete("images_collection", to_delete)


async def fetch_paper(arxiv_id, q, session, sem) -> None:
    url = "https://export.arxiv.org/pdf/%s.pdf" % arxiv_id

    try:
        async with sem:
            async with session.get(url) as resp:
                path = get_paper_path(arxiv_id)
                
                total = int(resp.headers.get('content-length', 0))
                
                with open(path, "wb") as f, tqdm(total=total, unit='B', unit_scale=True, desc=arxiv_id) as bar:
                    async for chunk in resp.content.iter_chunked(1024):
                        f.write(chunk)
                        bar.update(len(chunk))
                    
                q.put(arxiv_id)
                logging.info("fetched %s" % arxiv_id)
                
    except Exception as e:
        logging.warning("exception during fetching %s" % e)


def process_papers(q: Queue, extractor, vectorizer, idb, edb, num_ids) -> None:
    last_id = max(map(int, idb.keys()), default=0) + 1
    processed = 0
    
    with tqdm(total=num_ids, desc="processed arxiv ids") as bar:
        while True:
            arxiv_id = q.get()

            if arxiv_id is None:
                q.task_done()
                break
            
            try:
                base_id, version = split_id(arxiv_id)
                path = get_paper_path(arxiv_id)

                out = extractor(path, verbose=False)
                
                if not out:
                    logging.info("no figures found in %s. skipping..." % arxiv_id)
                    continue
                
                figures, captions = zip(*out)
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
                logging.info("processed and embedded %d figures for %s" % (nfig, arxiv_id))

                last_id += nfig
                processed += nfig
                
            except Exception as e:
                logging.warning("Exception while processing %s: %s" % (arxiv_id, e))
            
            os.remove(path)
            bar.update()
            q.task_done()
        
    logging.info("done! processed and added %d figures" % processed)


async def fetch_papers(arxiv_ids, q, max_concurrency) -> None:
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

    non_matching_ids = get_non_matching_ids(pdb, idb)
    logging.info("found %d non-matching ids" % len(non_matching_ids))
    
    arxiv_ids = non_matching_ids[:args.num]
    delete_ids(edb, idb, arxiv_ids)

    q = Queue()
    
    for file in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, file))
    
    extractor, vectorizer = get_models()
    
    t = threading.Thread(target=process_papers, 
                         args=(q, extractor, vectorizer, idb, edb, len(arxiv_ids)))
    t.start()

    asyncio.run(fetch_papers(arxiv_ids, q, max_concurrency=3))

    q.put(None)
    t.join()


if __name__ == "__main__":
    main()
