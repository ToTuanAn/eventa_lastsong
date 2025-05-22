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

for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    if row["chunk_emb"] != "":
        continue
    try:
        article_group_df = data_df[
            data_df["article_id"] == row["article_id"]
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

        # Perform chunked pooling
        embeddings = chunked_pooling(model_output, [span_annotations])[0]
        assert len(chunks) == len(chunk_contents) == len(embeddings), f"Length mismatch: {len(chunks)}, {len(chunk_contents)}, {len(embeddings)}"
        
        # Save the embeddings of each chunk into the dataframe
        for i, chunk in enumerate(chunks):
            data_df.loc[
                (data_df["article_id"] == row["article_id"])
                & (data_df["chunk_index"] == article_group_df.iloc[i]["chunk_index"]),
                "chunk_emb"
            ] = json.dumps(embeddings[i].tolist())

        if idx % 500 == 0:
            data_df.to_parquet(f"../processed_data/{db_name}_chunk_emb.parquet")

    except Exception as e:
        print(f"Error processing article_id {row['article_id']}: {e}; row {idx}")
        print(traceback.print_exc())
        continue


data_df.to_parquet(f"../processed_data/{db_name}_chunk_emb.{ext}", engine="auto")
