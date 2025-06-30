import pickle

import polars as pl
from qdrant_client import QdrantClient, models
from tqdm import tqdm

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)

collection_name = "news"

chunks_df = (
    pl.read_parquet("database_v4.parquet")
    .unique(["article_id", "chunk_index"])
    .sort(["article_id", "chunk_index"])
)

with open("jina_embeddings_v1.pkl", "rb") as f:
    jina_embeds = pickle.load(f)


batch_size = 512
total = []
for i in tqdm(range(0, len(chunks_df), batch_size), total=len(chunks_df) // batch_size):
    batch = []
    for i in range(i, i + batch_size):
        if i >= len(chunks_df):
            break
        point = models.PointVectors(
            id=chunks_df[i, "chunk_id"],
            vector={
                "jinaai/jina-embeddings-v3": jina_embeds[i],
            },
        )
        batch.append(point)

    qdrant_client.update_vectors(collection_name=collection_name, points=batch)
