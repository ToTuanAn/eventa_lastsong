import os
import pickle
from typing import Iterable

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
import polars as pl
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import AutoTokenizer

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)

## Embedding
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
converter = DocumentConverter()

# database_df = pl.read_json("data/database/database.json")
# database_df = database_df.transpose(
#     include_header=True, header_name="id", column_names=["article"]
# ).unnest("article")
train_df = pl.read_csv("data/Train Set/gt_train.csv")


def get_embeddings(documents: Iterable[str]):
    dense_embeddings = list(dense_embedding_model.embed(documents))
    bm25_embeddings = list(bm25_embedding_model.embed(documents))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(documents))

    return dense_embeddings, bm25_embeddings, late_interaction_embeddings


with open("results_v3.pkl", "rb") as f:
    results_df = pl.DataFrame(pickle.load(f))

results_df.head()
# ## Retrieval
# query = train_df[5, "caption"]
# id = train_df[5, "retrieved_article_id"]
# dense_vectors = next(dense_embedding_model.query_embed(query))
# sparse_vectors = next(bm25_embedding_model.query_embed(query))
# late_vectors = next(late_interaction_embedding_model.query_embed(query))
index = 39
article_id = train_df[index, "retrieved_article_id"]
caption = train_df[index, "caption"]
dense_vectors = results_df[index, "dense_embedding"]
sparse_vectors = results_df[index, "sparse_embedding"]
late_vectors = results_df[index, "late_interaction_embedding"]

prefetch = [
    models.Prefetch(
        query=dense_vectors,
        using="all-MiniLM-L6-v2",
        limit=20,
    ),
    models.Prefetch(
        query=models.SparseVector(**sparse_vectors.as_object()),
        using="bm25",
        limit=20,
    ),
]

## Rerank
rerank_points = qdrant_client.query_points(
    "hybrid-search",
    prefetch=prefetch,
    query=late_vectors,
    using="colbertv2.0",
    with_payload=True,
    limit=10,
)

dense_points = qdrant_client.query_points(
    "hybrid-search",
    query=dense_vectors,
    using="all-MiniLM-L6-v2",
    with_payload=True,
    limit=10,
)

points = qdrant_client.query_points(
    "hybrid-search",
    query=models.SparseVector(**sparse_vectors.as_object()),
    using="bm25",
    with_payload=True,
    limit=10,
)

print("aricle_id", article_id)
for point in points.points:
    print(
        f"{point.payload['id']} {point.score} {'here' if article_id == point.payload['id'] else ''}"
    )


def evaluate_retrieval(documents: Iterable[dict], batch_size: int = 256):
    # iterate over batch of documents wibatch_sizeuments

    results = []
    for idx in tqdm(range(0, len(documents), batch_size)):
        batch_documents = documents[idx : idx + batch_size]
        dense_embeddings, sparse_embeddings, late_interaction_embeddings = get_embeddings(
            batch_documents["caption"]
        )

        for dense_embedding, sparse_embedding, late_interaction_embedding, doc in zip(
            dense_embeddings,
            sparse_embeddings,
            late_interaction_embeddings,
            batch_documents.iter_rows(named=True),
        ):
            prefetch = [
                models.Prefetch(
                    query=dense_embedding,
                    using="all-MiniLM-L6-v2",
                    limit=10,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_embedding.as_object()),
                    using="bm25",
                    limit=10,
                ),
            ]
            points = qdrant_client.query_points(
                "hybrid-search",
                prefetch=prefetch,
                query=late_interaction_embedding,
                using="colbertv2.0",
                with_payload=True,
                limit=2,
            )

            results.append(
                {
                    "image_index": doc["image_index"],
                    "image_id": doc["retrieved_image_id"],
                    "article_id": doc["retrieved_article_id"],
                    "retrieved_article_id": points.points[0].payload["id"],
                    "index": points.points[0].payload["index"],
                    "point_id": points.points[0].id,
                    "score": points.points[0].score,
                    "dense_embedding": dense_embedding,
                    "sparse_embedding": sparse_embedding,
                    "late_interaction_embedding": late_interaction_embedding,
                }
            )

    return results


results = evaluate_retrieval(train_df, batch_size=16)

results_df = pl.DataFrame(results)
accuracy = (results_df["article_id"] == results_df["retrieved_article_id"]).mean()
print(f"Accuracy: {accuracy}")

with open("results_v3.pkl", "wb") as f:
    pickle.dump(results_df.to_dicts(), f, protocol=pickle.HIGHEST_PROTOCOL)

with open("results_v3.pkl", "rb") as f:
    results_df = pl.DataFrame(pickle.load(f))

results_df.head()
