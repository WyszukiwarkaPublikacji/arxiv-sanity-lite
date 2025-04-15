import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from pymilvus import AnnSearchRequest, WeightedRanker

IMAGE_MODEL = "openai/clip-vit-base-patch16"
CAPTION_MODEL = "all-MiniLM-L6-v2"


class FigureVectorizer:
    def __init__(self, device):
        self.caption_vectorizer = SentenceTransformer(CAPTION_MODEL, device=device)
        self.image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer = CLIPModel.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer.to(device)

        self.device = device

    def __call__(self, captions, images=None, batch_size=32):
        caption_emb = self.caption_vectorizer.encode(
            captions,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).cpu().tolist()

        img_emb = []
        if images:
            for idx in range(0, len(images), batch_size):
                batch = images[idx : idx + batch_size]

                inputs = self.image_processor(
                    images=batch, return_tensors="pt", padding=True, size=256
                )

                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with torch.no_grad():
                    output = self.image_vectorizer.get_image_features(**inputs).cpu()
                    img_emb.extend(output.tolist())

        return img_emb, caption_emb


def hybrid_search(client, image_embedding, caption_embedding, limit=10):
    image_req = AnnSearchRequest(
        data=image_embedding, 
        anns_field="image_embedding", 
        limit=limit, 
        param={'nprobe': 10}
    )

    caption_req = AnnSearchRequest(
        data=caption_embedding, 
        anns_field="caption_embedding", 
        limit=limit, 
        param={'nprobe': 10}
    )

    results = client.hybrid_search(
        collection_name="images_collection",
        reqs=[image_req, caption_req],
        ranker=WeightedRanker(0.6, 0.4),
        limit=limit,
    )

    return results
