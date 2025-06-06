from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig
import torch
from src.model.image_emb.visual_bge.visual_bge.modeling import Visualized_BGE


class BgeVisualizedModel:
    def __init__(self, config) -> None:
        self.model = Visualized_BGE(model_name_bge =config["model"]["name"], model_weight=config["model"]["model_weight"]).eval()
    
    def embed(self, text=None, image_file=None):
        with torch.no_grad():
            if text is not None and image_file is not None:
                query_emb = self.model.encode(image=image_file, text=text)
            elif text is not None:
                query_emb = self.model.encode(text=text)
            elif image_file is not None:
                query_emb = self.model.encode(image=image_file)
            else:
                query_emb = None 
        
        return query_emb

