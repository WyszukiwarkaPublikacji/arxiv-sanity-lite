import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Thread

import fitz
import numpy as np
from PIL import Image
import logging
from aslite import config

from image_search import get_paper_path


CAPTION_REGEX = re.compile(
    r"\b(fig(?:(?:ure)|.)?\s*\S+)\s*[:.,|]?\s+(\S+[\s\S]+)", re.IGNORECASE
)


def render_page(page, dpi):
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.asarray(image)


def render_arxiv_id(arxiv_id, ensure_captions, dpi):
    try:
        path = get_paper_path(arxiv_id)
        pdf = fitz.open(path)
        
        pages_data = list()

        for page in pdf:
            blocks = list()
            
            for block in page.get_text("blocks"):
                *bbox, t = block[:5]

                if not ensure_captions:
                    blocks.append([t, bbox])
                    continue

                if re.match(CAPTION_REGEX, t):
                    blocks.append([t, bbox])
            
            if blocks:
                render = render_page(page, dpi=dpi)
                pages_data.append((blocks, render))

        return arxiv_id, pages_data
    
    except Exception as e:
        logging.warning("exception during rendering: %s" % e)
        return arxiv_id, []

class PageStream:
    def __init__(
        self,
        queue: Queue,
        batch_size: int = 1,
        ensure_captions: bool = False,
        max_workers: bool = None,
        dpi: int = 150,
    ):
        self.queue = queue
        self.batch_size = batch_size
        self.ensure_captions = ensure_captions
        self.max_workers = max_workers or os.cpu_count()
        self.dpi = dpi
        
        self.batch_buffer = Queue(maxsize=8)

    def producer(self):
        futures = Queue()

        def submitter():
            nonlocal futures
            count = 0

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for arxiv_id in iter(self.queue.get, None):
                    f = executor.submit(
                        render_arxiv_id,
                        arxiv_id = arxiv_id,
                        ensure_captions=self.ensure_captions,
                        dpi=self.dpi
                    )
                    futures.put(f)
                        
                futures.put(None)

        t = Thread(target=submitter, daemon=True)
        t.start()

        batch = list()
        active_futures = set()
        collected_ids = set()

        while True:
            while not futures.empty():
                f = futures.get()

                if f is None:
                    break

                active_futures.add(f)
                futures.task_done()

            if not active_futures:
                if futures.empty() and not t.is_alive():
                    break

                continue

            for f in as_completed(active_futures):
                arxiv_id, pages_data = f.result()
                collected_ids.add(arxiv_id)

                for render, blocks in pages_data:
                    batch.append((arxiv_id, render, blocks))
                    
                    if len(batch) >= self.batch_size:
                        self.batch_buffer.put((collected_ids, batch))
                        batch = list()
                        collected_ids = set()

                active_futures.remove(f)

        if batch:
            self.batch_buffer.put((collected_ids, batch))
            
        self.batch_buffer.put(None)
            
        t.join()

    def __iter__(self):
        t = Thread(target=self.producer, daemon=True)
        t.start()
        
        for collected_ids, batch in iter(self.batch_buffer.get, None):
            yield collected_ids, batch
            
        t.join()