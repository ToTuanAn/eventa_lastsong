import re
import pandas as pd
import json
import pickle
from tqdm import tqdm
import traceback

from transformers import AutoModel, AutoTokenizer
import torch

from chunked_pooling import (
    chunked_pooling,
    chunk_by_sentences,
    chunk_by_config,
    chunk_by_config2,
)
from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device_map=device
)
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device_map=device
)

task_chunked_retrieval = AbsTaskChunkedRetrieval(
    chunking_strategy="semantic",
    long_late_chunking_embed_size=model.config.max_position_embeddings
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

_i = 0

for idx in tqdm(range(len(data_df))):
    row = data_df.iloc[idx]
    if row["chunk_emb"] != "":
        # print(f"Skipping already processed row {idx} for article_id {row['article_id']}")
        continue
    try:
        _i += 1
        article_group_df = data_df[
            data_df["article_id"] == row["article_id"]
        ].sort_values("chunk_index")

        chunk_contents = article_group_df["chunk_content"].tolist()
        chunk_contents = [
            process_str(x) + "." for x in chunk_contents
        ]

        article_context = " ".join(chunk_contents)

        inputs = tokenizer(article_context, return_tensors="pt")

        # Check if input exceeds max length
        # if inputs["input_ids"].shape[1] < model.config.max_position_embeddings:
        #     # print(f"Input exceeds max length for article_id {row['article_id']}. Skipping.")
        #     continue
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # print(f"Processing article_id {row['article_id']} with {len(chunk_contents)} chunks")

        chunks, span_annotations = chunk_by_config2(
            article_context, tokenizer, chunk_contents
        )
        
        with torch.no_grad():                
            if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
                model_outputs = task_chunked_retrieval._embed_with_overlap(model, inputs)
                output_embs = chunked_pooling(
                    [model_outputs], [span_annotations], max_length=None
                )
            else:
                model_outputs = model(**inputs)
                output_embs = chunked_pooling(
                    model_outputs,
                    [span_annotations],
                    max_length=model.config.max_position_embeddings,
                )

        embeddings = output_embs[0]

        # Perform chunked pooling
        # embeddings = chunked_pooling([model_outputs], [span_annotations])[0]
        assert len(chunks) == len(chunk_contents) == len(embeddings), f"Length mismatch: {len(chunks)}, {len(chunk_contents)}, {len(embeddings)}"
        
        # Save the embeddings of each chunk into the dataframe
        for i, chunk in enumerate(chunks):
            idx_to_update = article_group_df.index[i]
            data_df.at[idx_to_update, "chunk_emb"] = json.dumps(embeddings[i].tolist())

        #     print(f"Processed chunk {i+1}/{len(chunks)} for article_id {row['article_id']}")
        # print(f"Updating vectors for article_id {row['article_id']} with {len(embeddings)} embeddings.")

        # qdrant_client.update_vectors(
        #     collection_name=collection_name,
        #     points=[
        #         models.PointVectors(
        #             id=id,  # chunk id
        #             vector={
        #                 "jinaai/jina-embeddings-v2-base-en": embed,
        #             },
        #         ) for id, embed in zip(article_group_df["chunk_id"].tolist(), embeddings)
        #     ],
        # )

        if _i % 1000 == 0:
            data_df.to_parquet(f"../processed_data/{db_name}_chunk_emb.parquet")

    except Exception as e:
        print(f"Error processing article_id {row['article_id']}: {e}; row {idx}")
        print(traceback.print_exc())
        continue


data_df.to_parquet(f"../processed_data/{db_name}.{ext}", engine="auto")
