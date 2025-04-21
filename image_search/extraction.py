import logging
import os

logger = logging.getLogger(__name__)

import ultralytics

import numpy as np
import requests
from tqdm import tqdm


WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=12wkHnhD49uCoBIEUrJ_IVHpHutu0qNCe"


def box_center(box):
    return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2


def crop(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def distance_matrix(a, b, wx=1.0, wy=1.0, w_above=1.0):
    a, b = np.array(a), np.array(b)
    dists = (a[:, None] - b[None, :]) ** 2

    dx = dists[..., 0] * wx
    dy = dists[..., 1] * wy

    # penalize captions above
    mask = b[:, 1] < a[:, None, 1]
    dy[mask] *= w_above

    return dx + dy


class FigureExtractor:
    def __init__(self, weights_path="./data/weights/weights.pt"):
        self.model = self._load_model(weights_path)

    def _download_weights(self, path):
        logger.info("downloading weights...")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with requests.get(WEIGHTS_URL, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))

            with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))

    def _load_model(self, weights_path, fresh=False):
        """load model and download weights if not found"""
        if fresh:
            os.remove(weights_path)

        try:
            if not os.path.exists(weights_path):
                self._download_weights(weights_path)

            model = ultralytics.YOLO(weights_path)

        except RuntimeError as e:
            logger.warning(e)
            logger.info("reloading model...")
            return self._load_model(weights_path, fresh=True)

        return model    

    def _match_captions(self, render, captions, cap_centers, bboxes):
        bboxes = [box.astype(int) for box in bboxes]

        if bboxes:
            fig_centers = [box_center(box) for box in bboxes]

            # x-axis distance is less important
            # texts above the figure have an additional penalty
            dists = distance_matrix(fig_centers, cap_centers, wx=0.5, w_above=1.5)
            idx = np.argmin(dists, axis=1)

            return ((crop(render, box), captions[i]) for i, box in zip(idx, bboxes))

    def __call__(self, ids, blocks_batch, renders, **kwargs):
        out = []
        results = self.model.predict(renders, **kwargs)

        for id, render, blocks, result in zip(ids, renders, blocks_batch, results):
            bboxes = [box.cpu().numpy() for box in result.boxes.xyxy]
            text, block_bboxes = zip(*blocks)
            block_centers = [box_center(bb) for bb in block_bboxes]

            m = self._match_captions(render, text, block_centers, bboxes)
            if m:
                out.extend((id, cap, fig) for fig, cap in m)

        return out
