import re
import pandas as pd
import json
import pickle
from tqdm import tqdm
import concurrent.futures
import traceback

from transformers import AutoModel, AutoTokenizer
import torch

from chunked_pooling import (
    chunked_pooling,
    chunk_by_sentences,
    chunk_by_config,
    chunk_by_config2,
)

from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"  # "https://a836-1-53-255-146.ngrok-free.app"

qdrant_client = QdrantClient(url=QDRANT_URL, timeout=1000)
collection_name = "news"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device_map=device
)
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device_map=device
)

def process_str(s):
    """
    Process the input string by removing newlines and extra spaces
    :param str: The input string to process
    :return: The processed string
    """
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()

# put model in eval mode and use no_grad context
model.eval()

db_name = "database_v4"
ext = "parquet"

data_df = pd.read_parquet(f"..\\processed_data\\{db_name}.{ext}", engine="auto")
data_df = data_df.reset_index(drop=True)
data_df["row_number"] = data_df.index
data_df["chunk_emb"] = ["" for _ in range(len(data_df))]

def process_article_group(article_id):
    try:
        article_group_df = data_df[
            data_df["article_id"] == article_id
        ].sort_values("chunk_index")

        chunk_contents = article_group_df["chunk_content"].tolist()
        chunk_contents = [
            process_str(x) + "." for x in chunk_contents
        ]

        article_context = " ".join(chunk_contents)

        chunks, span_annotations = chunk_by_config2(
            article_context, tokenizer, chunk_contents
        )

        inputs = tokenizer(article_context, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model_output = model(**inputs)

        embeddings = chunked_pooling(model_output, [span_annotations])[0]
        assert len(chunks) == len(chunk_contents) == len(embeddings), f"Length mismatch: {len(chunks)}, {len(chunk_contents)}, {len(embeddings)}"

        # Prepare Qdrant update
        qdrant_client.update_vectors(
            collection_name=collection_name,
            points=[
                models.PointVectors(
                    id=id,  # chunk id
                    vector={
                        "jinaai/jina-embeddings-v2-base-en": embed,
                    },
                ) for id, embed in zip(article_group_df["chunk_id"].tolist(), embeddings)
            ],
        )
        return article_id
    except Exception as e:
        print(f"Error processing article_id {article_id}: {e};")
        print(traceback.print_exc())
        return article_id
    

unique_article_ids = data_df["article_id"].unique().tolist()
print(f"Total unique article_ids: {len(unique_article_ids)}")

# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     try:
#         futures = []
#         unique_article_ids = data_df["article_id"].unique().tolist()
#         for article_id in unique_article_ids:
#             futures.append(executor.submit(process_article_group, article_id))

#         for i, future in enumerate(concurrent.futures.as_completed(futures)):
#             article_id = future.result()
#             print(f"Processed article_id {article_id}")
#             # if embeddings is not None:
#             #     data_df.at[row_number, "chunk_emb"] = json.dumps([emb.tolist() for emb in embeddings])
#             # if i % 500 == 0:
#             #     data_df.to_parquet(f"..\\processed_data\\{db_name}_chunk_emb.parquet")
#     except KeyboardInterrupt:
#         print("Keyboard interrupt detected. Exiting...")

# data_df.to_parquet(f"..\\processed_data\\{db_name}_chunk_emb.{ext}", engine="auto")