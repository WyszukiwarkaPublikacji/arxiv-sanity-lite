import torch
from sentence_transformers import SentenceTransformer
from transformers import DonutProcessor, VisionEncoderDecoderModel
# from transformers import CLIPProcessor, CLIPModel

# IMAGE_MODEL = "openai/clip-vit-base-patch32"
CHART_MODEL = "ahmed-masry/unichart-base-960"
TEXT_MODEL = "all-MiniLM-L6-v2"
# TEXT_MODEL = "paraphrase-MiniLM-L6-v2"


class FigureVectorizer:
    def __init__(self, device):
        self.device = device
        self.text_vectorizer = SentenceTransformer(TEXT_MODEL, device=device)
        
        self.chart_processor = DonutProcessor.from_pretrained(CHART_MODEL)
        self.chart_vectorizer = VisionEncoderDecoderModel.from_pretrained(CHART_MODEL)
        self.chart_vectorizer.to(self.device)

    def text_embedding(self, text, batch_size=64):        
        embedding = self.text_vectorizer.encode(
            text,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        
        embedding /= torch.norm(embedding, p=2, dim=-1, keepdim=True)
        return embedding
    
    def image_embedding(self, images, batch_size=32):         
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
        
    def chart_embedding(self, images, batch_size=32, input_size=960):        
        embeddings = []
        
        decoder_input_ids = self.chart_processor.tokenizer(
            '<summarize_chart> <s_answer>', 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        # could be moved into the loop to save gpu memory
        images = self.chart_processor(
            images, 
            return_tensors="pt", 
            size={"height": input_size, "width": input_size}
        ).pixel_values.to(self.device)
        
        for idx in range(0, len(images), batch_size):
            pixel_values = images[idx : idx + batch_size]
            input_ids = decoder_input_ids.expand(pixel_values.size(0), -1)
            
            with torch.no_grad():
                outputs = self.chart_vectorizer(
                    pixel_values=pixel_values,
                    decoder_input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            embedding = outputs.decoder_hidden_states[-1].mean(1)
            embeddings.append(embedding)
    
        embeddings = torch.vstack(embeddings)
        embeddings /= torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings
        
    def __call__(self, captions, images, batch_size=32, input_size=960):
        caption_embeddings = self.text_embedding(captions, batch_size).cpu()
        chart_embeddings = self.chart_embedding(images, batch_size, input_size).cpu() 
        
        return caption_embeddings.tolist(), chart_embeddings.tolist()
