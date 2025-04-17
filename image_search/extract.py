import logging
import os

logger = logging.getLogger(__name__)

import re

import ultralytics
import io

import fitz
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

CAPTION_REGEX = re.compile(r"\b(fig(?:ure)?)\s*([\s\S]+)?", re.IGNORECASE)

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


def render_page(page, dpi=100):
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image = np.asarray(image)
    return image


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

    def _match_captions(self, render, captions, caption_centers, bboxes):
        bboxes = [box.astype(int) for box in bboxes]

        if bboxes:
            bbox_centers = [box_center(box) for box in bboxes]

            # x-axis distance is less important
            # texts above the figure have an additional penalty
            dists = distance_matrix(bbox_centers, caption_centers, wx=0.5, w_above=1.5)

            indices = np.argmin(dists, axis=1)

            return [
                (crop(render, bbox), captions[idx])
                for bbox, idx in zip(bboxes, indices)
            ]

        # return [(crop(render, box), None) for box in bboxes]
        return []

    def __call__(self, path, **kwargs):
        out = []
        pdf = fitz.open(path)

        all_captions = [
            [b for b in page.get_text("blocks") if re.match(CAPTION_REGEX, b[4])]
            for page in pdf
        ]

        renders = [render_page(p, dpi=100) for p, c in zip(pdf, all_captions) if c]

        results = self.model.predict(renders, **kwargs)

        for render, caption_blocks, result in zip(
            renders, filter(None, all_captions), results
        ):
            bboxes = [box.cpu().numpy() for box in result.boxes.xyxy]
            captions = [c[4] for c in caption_blocks]
            centers = [box_center(c[:4]) for c in caption_blocks]

            out.extend(self._match_captions(render, captions, centers, bboxes))

        return out
