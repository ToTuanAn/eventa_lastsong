from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig
import torch

class ColQwen25Model:
    def __init__(self, config):
        self.processor = ColQwen2_5_Processor.from_pretrained(config["processor"]["name"])

        if "peft_path" in config["model"] and config["model"]["peft_path"] is not None:
            print(f"Loading model {config["model"]["name"]}")
            print(f"Loading PEFT {config["model"]["peft_path"]}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = PeftModel.from_pretrained(ColQwen2_5.from_pretrained(
                                                        config["model"]["name"],
                                                        torch_dtype=torch.bfloat16,
                                                        quantization_config=bnb_config,
                                                    ),
                                                    config["model"]["peft_path"]).to("cuda:0").eval()
        else:
            print(f"Loading model {config["model"]["name"]}")
            self.model = ColQwen2_5.from_pretrained(
                config["model"]["name"],
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
            ).eval()


    def get_patches(self, image_size):
        return self.processor.get_n_patches(image_size,
                                            spatial_merge_size=self.model.spatial_merge_size)

    def get_query_embedding(self, query):
        processed_queries = self.processor.process_queries([query]).to(self.model.device)

        # Resulting query embedding is a tensor of shape (22, 128)
        query_embedding = self.model(**processed_queries)[0]
        return query_embedding


    def embed_and_mean_pool_batch(self, image_batch):
        #embed
        with torch.no_grad():
            processed_images = self.processor.process_images(image_batch).to(self.model.device)
            image_embeddings = self.model(**processed_images)


        image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

        #mean pooling
        pooled_by_rows_batch = []
        pooled_by_columns_batch = []


        for image_embedding, tokenized_image, image in zip(image_embeddings,
                                                           processed_images.input_ids,
                                                           image_batch):
            x_patches, y_patches = self.get_patches(image.size)
            #print(f"{model_name} model divided this PDF page in {x_patches} rows and {y_patches} columns")

            image_tokens_mask = (tokenized_image == self.processor.image_token_id)

            image_tokens = image_embedding[image_tokens_mask].view(x_patches, y_patches, self.model.dim)
            pooled_by_rows = torch.mean(image_tokens, dim=0)
            pooled_by_columns = torch.mean(image_tokens, dim=1)

            image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
            first_image_token_idx = image_token_idxs[0].cpu().item()
            last_image_token_idx = image_token_idxs[-1].cpu().item()

            prefix_tokens = image_embedding[:first_image_token_idx]
            postfix_tokens = image_embedding[last_image_token_idx + 1:]

            #print(f"There are {len(prefix_tokens)} prefix tokens and {len(postfix_tokens)} in a {model_name} PDF page embedding")

            #adding back prefix and postfix special tokens
            pooled_by_rows = torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0).cpu().float().numpy().tolist()
            pooled_by_columns = torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0).cpu().float().numpy().tolist()

            pooled_by_rows_batch.append(pooled_by_rows)
            pooled_by_columns_batch.append(pooled_by_columns)


        return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch
