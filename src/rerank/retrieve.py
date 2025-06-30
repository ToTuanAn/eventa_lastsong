import json

import polars as pl
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import AutoModel

from fastembed import SparseTextEmbedding


QDRANT_URL = "http://127.0.0.0:6333"
QDRANT_API_KEY = ""

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)
qdrant_client.get_collections()
collection_name = "news"

data = "../../data"
test_df = pl.read_csv(f"{data}/raw/private_test/query.csv").rename(
    {"query_text": "caption"}
)

bm42_embedding_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

jina_embedding_v3_model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    device_map="cuda:0",
)

jina_embedding_v3_model.encode(test_df[0, "caption"])

data = test_df["caption"]

embedded_query = dict(sparse_embedding=[], dense_embedding=[])

batch_size = 64
for idx in tqdm(range(0, len(data), batch_size)):
    batch_queries = data[idx : idx + batch_size]

    embedded_query["sparse_embedding"].extend(
        bm42_embedding_model.embed(batch_queries.to_list())
    )

    embedded_query["dense_embedding"].extend(
        list(
            jina_embedding_v3_model.encode(
                batch_queries.to_list(), task="retrieval.query"
            )
        )
    )


# %%
query_points = []
for sparse_embedding, dense_embedding in tqdm(
    zip(embedded_query["sparse_embedding"], embedded_query["dense_embedding"]),
    total=len(embedded_query["sparse_embedding"]),
):
    point = []
    point.extend(
        qdrant_client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(**sparse_embedding.as_object()),
            using="Qdrant/bm42-all-minilm-l6-v2-attentions",
            with_payload=True,
            limit=20,
        ).points
    )

    point.extend(
        qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_embedding,
            using="jinaai/jina-embeddings-v3",
            with_payload=True,
            limit=20,
        ).points
    )
    query_points.append(point)


# %%
article_list = []
article_score = []
for points in query_points:
    article_list.append(json.dumps([point.payload["article_id"] for point in points]))
    article_score.append(json.dumps([point.score for point in points]))

# %%
test_df = test_df.with_columns(
    article_list=pl.Series(article_list),
    article_score=pl.Series(article_score),
)

test_df.write_csv(
    "private_test_bm42_jina_v3_query_result_top_20.csv",
)

# %%
