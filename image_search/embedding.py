import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


IMAGE_MODEL = "all-MiniLM-L6-v2"
CAPTION_MODEL = "openai/clip-vit-base-patch16"


class FigureVectorizer:
    def __init__(self, device):
        self.caption_vectorizer = SentenceTransformer(CAPTION_MODEL, device=device)
        self.image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer = CLIPModel.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer.to(device)

        self.device = device

    def __call__(self, images, captions, batch_size=32):
        caption_emb = self.caption_vectorizer.encode(
            captions, batch_size=batch_size, convert_to_tensor=True
        ).cpu()

        img_emb = []
        for idx in range(0, len(images), batch_size):
            batch = images[idx : idx + batch_size]

            inputs = self.image_processor(
                images=batch, return_tensors="pt", padding=True, size=256
            )

            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                output = self.image_vectorizer.get_image_features(**inputs).cpu()
                img_emb.append(output)

        img_emb = torch.vstack(img_emb)

        caption_emb /= caption_emb.norm(p=2, dim=-1, keepdim=True)
        img_emb /= img_emb.norm(p=2, dim=-1, keepdim=True)

        out = torch.cat([caption_emb, img_emb], dim=1)
        out /= out.norm(p=2, dim=-1, keepdim=True)
        return out
