import argparse
import asyncio
import logging
import os
import aiohttp

from aslite.db import get_embeddings_db, get_images_db, get_papers_db
from image_search import PageStream, load_models, get_image_path, get_paper_path

from queue import Queue
import threading
from tqdm.asyncio import tqdm

import cv2
from aslite import config


def _arxiv_id(base_id, version) -> str:
    if version > 0:
        base_id += f"v{version}"
    return base_id


def split_id(aid) -> tuple[str, int]:
    aid = aid.split("v")
    version = int(aid[-1]) if aid[-1].isdigit() else 0
    return aid[0], version


def get_last_id(db):
    return max(map(int, db.keys()), default=-1) + 1


def remove_pdfs_by_ids(ids):
    for id in ids:
        path = get_paper_path(id)
        if os.path.exists(path):
            os.remove(path)
    

def non_matching_ids(pdb, idb) -> list[str]:
    """Find all arxiv ids in papers.db that are either missing
    in data/images.db or have a newer version available."""
    non_matching_ids = set()
    base_ids = {k: v for k, v in map(split_id, (el for el in pdb.keys() if pdb[el]["provider"] == "arxiv"))}

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


def delete_ids(client, idb, arxiv_ids) -> None:
    to_delete = []
    base_ids = set(split_id(aid)[0] for aid in arxiv_ids)

    for id, info in idb.items():
        if info["base_id"] in base_ids:
            to_delete.append(id)
            del idb[id]
            os.remove(get_image_path(id))

    if to_delete:
        client.delete("images_collection", to_delete)
        
        
def save_figures(idb, ids, data):    
    for id, (arxiv_id, caption, figure) in zip(ids, data):
        base_id, version = split_id(arxiv_id)
        fig_path = get_image_path(id)
        cv2.imwrite(fig_path, figure[..., ::-1])

        idb[id] = dict(base_id=base_id, version=version, caption=caption)


def data_dicts(ids, caption_embeddings, image_embeddings):
    return [
        dict(id=id, image_embedding=x, caption_embedding=y) for
        id, x, y in zip(ids, image_embeddings, caption_embeddings)
    ]


async def fetch_paper(arxiv_id, q, session, sem, bar) -> None:
    url = "https://export.arxiv.org/pdf/%s.pdf" % arxiv_id

    try:
        async with sem:
            async with session.get(url) as resp:
                path = get_paper_path(arxiv_id)

                with open(path, "wb") as f:
                    f.write(await resp.read())

                q.put(arxiv_id)
                bar.update()

    except Exception as e:
        logging.warning("exception during fetching %s" % e)


async def fetch_papers(arxiv_ids, q, max_concurrency) -> None:
    sem = asyncio.Semaphore(max_concurrency)
    bar = tqdm(total=len(arxiv_ids), desc="fetched")

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_paper(id, q, session, sem, bar) for id in arxiv_ids]
        await asyncio.gather(*tasks)
        
    bar.close()


def process_papers(q: Queue, extractor, vectorizer, idb, edb, total_ids) -> None:        
    last_id = get_last_id(idb)
    stream = PageStream(
        q, 
        batch_size=config.extraction_batch_size, 
        min_caption_length=config.rendering_min_caption_length,
        dpi=config.rendering_dpi
    )
    
    # keep track of processed ids - some can appear in more than one batch
    processed_ids = set()

    for collected_ids, batch in stream:
        try:
            arxiv_ids, renders, blocks = zip(*batch)
            out = extractor(arxiv_ids, renders, blocks, verbose=False)
            
            # when we receive an arxiv_id we can be sure it was 
            # processed even if it appear in more than one batch
            remove_pdfs_by_ids(collected_ids)
            processed_ids.update(collected_ids)

            if not out:
                logging.info("no figures found. skipping...")
                continue
            
            new_ids = list(range(last_id, last_id + len(out)))
            save_figures(idb, new_ids, out)
            
            _, captions, figures = zip(*out)
            embeddings = vectorizer(
                captions, 
                figures, 
                input_size=config.model_input_size,
                batch_size=config.embedding_batch_size,
            )
            
            data = data_dicts(new_ids, *embeddings)
            edb.insert("images_collection", data)

            last_id += len(out)               
            
            logging.info("added %d figures" % len(out))
            logging.info("processed %d/%d files" % (len(processed_ids), total_ids))
                
        except Exception as e:
            logging.warning("Exception while processing: %s" % e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    parser = argparse.ArgumentParser(description="Extract images")
    parser.add_argument("-n", "--num", type=int, default=100, help="Up to how many papers to extract from")
    args = parser.parse_args()
    print(args)

    pdb = get_papers_db("r")
    idb = get_images_db("c")
    edb = get_embeddings_db()

    ids_to_update = non_matching_ids(pdb, idb)
    logging.info("found %d non-matching ids" % len(ids_to_update))

    ids_to_update = ids_to_update[:args.num]
    delete_ids(edb, idb, ids_to_update)

    q = Queue()    
    extractor, vectorizer = load_models()

    t = threading.Thread(
        target=process_papers, 
        args=(q, extractor, vectorizer, idb, edb, len(ids_to_update))
    )
    t.start()

    asyncio.run(fetch_papers(ids_to_update, q, max_concurrency=5))

    q.put(None)
    t.join()