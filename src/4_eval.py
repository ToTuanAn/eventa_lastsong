import json
import pickle

import polars as pl
from qdrant_client import QdrantClient
from tqdm import tqdm
from transformers import AutoModel

# QDRANT_URL = "https://bf00-1-53-255-146.ngrok-free.app/"
# QDRANT_API_KEY = ""

# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443, timeout=1000)
# collection_name = "news"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5000)

collection_name = "news"

data = "."
test_df = pl.read_csv(f"{data}/query.csv").rename({"query_text": "caption"})

jina_embedding_v2_model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    device_map="cuda:0",
)

# bm42_embedding_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

# jina_embedding_v2_model.encode(test_df[0, "caption"])

data = test_df["caption"]

embedded_query = dict(sparse_embedding=[])

batch_size = 64
for idx in tqdm(range(0, len(data), batch_size)):
    batch_queries = data[idx : idx + batch_size]

    embedded_query["sparse_embedding"].extend(
        list(jina_embedding_v2_model.encode(batch_queries.to_list(), task="retrieval.query"))
    )


# %%
jina_v3_points = []
bm42_points = []
for sparse_embedding in tqdm(embedded_query["sparse_embedding"]):
    bm42_points.append(
        qdrant_client.query_points(
            collection_name=collection_name,
            query=sparse_embedding,
            using="jinaai/jina-embeddings-v3",
            with_payload=True,
            limit=20,
        )
    )

    if len(bm42_points) % 300 == 0:
        with open("bm42_points.pkl", "wb") as f:
            pickle.dump(bm42_points, f)

    


# %%
article_list = []
article_score = []
for points in bm42_points:
    article_list.append(json.dumps([point.payload["article_id"] for point in points.points]))
    article_score.append(json.dumps([point.score for point in points.points]))

# %%
test_df = test_df.with_columns(
    article_list=pl.Series(article_list),
    article_score=pl.Series(article_score),
)

test_df.write_csv(
    "jina_v3_query_result_top_20.csv",
)

# %%
