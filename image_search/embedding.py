import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

IMAGE_MODEL = "openai/clip-vit-base-patch16"
TEXT_MODEL = "all-MiniLM-L6-v2"


class FigureVectorizer:
    def __init__(self, device):
        self.device = device
        self.text_vectorizer = SentenceTransformer(TEXT_MODEL, device=device)
        
        self.image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer = CLIPModel.from_pretrained(IMAGE_MODEL)
        self.image_vectorizer.eval().to(self.device)

    def text_embedding(self, text, batch_size=64):        
        embedding = self.text_vectorizer.encode(
            text,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        
        embedding /= torch.norm(embedding, p=2, dim=-1, keepdim=True)
        return embedding
        
    def image_embedding(self, images, batch_size=32, input_size=960):        
        embeddings = []
        
        for idx in range(0, len(images), batch_size):
            batch = images[idx : idx + batch_size]
            inputs = self.image_processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embedding = self.image_vectorizer.get_image_features(**inputs)

            embeddings.append(embedding)
    
        embeddings = torch.vstack(embeddings)
        embeddings /= torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings
        
    def __call__(self, captions, images, batch_size=32, input_size=960):
        caption_embeddings = self.text_embedding(captions, batch_size).cpu()
        image_embeddings = self.image_embedding(images, batch_size, input_size).cpu() 
        
        return caption_embeddings.tolist(), image_embeddings.tolist()
