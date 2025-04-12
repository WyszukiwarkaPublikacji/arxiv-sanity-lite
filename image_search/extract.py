import logging
import os

logger = logging.getLogger(__name__)

import re

import ultralytics

import fitz
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from pathlib import Path

CAPTION_REGEX = re.compile(r"\b(fig(?:ure)?)\s*([\s\S]+)?", re.IGNORECASE)

WEIGHTS_URL = (
    "https://drive.google.com/uc?export=download&id=1T5zWVVgcymGKtkyYREQ0-cxmbnBMnatX"
)


def box_center(box):
    return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2


def box_centers(boxes):
    return np.array([box_center(box) for box in boxes])


def crop(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def distance_matrix(a, b, wx=1.0, wy=1.0, w_above=1.0):
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
        engine_path = os.path.splitext(weights_path)[0] + ".engine"

        self.model = self._load_model(engine_path, weights_path)

    def _download_weights(self, path):
        logger.info("downloading weights...")
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        with requests.get(WEIGHTS_URL, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))

            with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))

    def _export_trt(self, model):
        logger.info("exporting to tensorrt format...")
        return model.export(format="engine", half=True)

    def _load_model(self, engine_path, weights_path, fresh=False):
        """load model and download weights if not found"""
        if fresh:
            os.remove(weights_path)
            os.remove(engine_path)

        try:
            if not os.path.exists(engine_path):
                if not os.path.exists(weights_path):
                    self._download_weights(weights_path)

                model = ultralytics.YOLO(weights_path)
                # engine_path = self._export_trt(model)

        except RuntimeError:
            return self._load_model(engine_path, weights_path, fresh=True)

        # model = ultralytics.YOLO(engine_path)
        return model

    def __call__(self, pdf, **kwargs):
        out = []
        
        pdf = fitz.open(pdf) if isinstance(pdf, str) else fitz.open(stream=pdf)

        images = [render_page(p, dpi=100) for p in pdf]
        results = self.model.predict(images, **kwargs)

        for page, image, result in zip(pdf, images, results):
            bboxes = [box.tolist() for box in result.boxes.xyxy]

            if bboxes:
                bboxes = np.array(bboxes, dtype=int)

                captions = []
                caption_centers = []

                for block in page.get_text("blocks"):
                    x1, y1, x2, y2, text, *_ = block
                    center_x, center_y = x2 - x1, y2 - y1

                    captions.append(text)
                    caption_centers.append((center_x, center_y))

                if captions:
                    caption_centers = np.array(caption_centers)
                    figure_centers = box_centers(bboxes)

                    # x-axis distance is less important
                    # texts above the figure should have an additional penalty
                    dists = distance_matrix(
                        figure_centers, caption_centers, wx=0.5, w_above=1.5
                    )

                    indices = np.argmin(dists, axis=1)

                    for bbox, idx in zip(bboxes, indices):
                        figure = crop(image, bbox)
                        caption = captions[idx]

                        out.append((figure, caption))

                else:
                    out.extend([(crop(image, bbox), None) for bbox in bboxes])

        return out

    def extract_from_dir(self, dir):
        figures, captions = [], []

        for file_name in os.listdir(dir):
            path = os.path.join(dir, file_name)
            new_figures, new_captions = zip(*self(path, verbose=False))

            figures.extend(new_figures)
            captions.extend(new_captions)

        return figures, captions
