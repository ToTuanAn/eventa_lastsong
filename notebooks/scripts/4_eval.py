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


# ## Retrieval
# query = train_df[5, "caption"]
# id = train_df[5, "retrieved_article_id"]
# dense_vectors = next(dense_embedding_model.query_embed(query))
# sparse_vectors = next(bm25_embedding_model.query_embed(query))
# late_vectors = next(late_interaction_embedding_model.query_embed(query))


def evaluate_point(index: int):
    article_id = train_df[index, "retrieved_article_id"]
    # caption = train_df[index, "caption"]
    dense_vectors = results_df[index, "dense_embedding"]
    sparse_vectors = results_df[index, "sparse_embedding"]
    late_vectors = results_df[index, "late_interaction_embedding"]

    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=5,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]

    dense_points = qdrant_client.query_points(
        "hybrid-search",
        query=dense_vectors,
        using="all-MiniLM-L6-v2",
        with_payload=True,
        limit=10,
    )

    sparse_points = qdrant_client.query_points(
        "hybrid-search",
        query=models.SparseVector(**sparse_vectors.as_object()),
        using="bm25",
        with_payload=True,
        limit=10,
    )

    late_points = qdrant_client.query_points(
        "hybrid-search",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=10,
    )

    print("aricle_id", article_id)
    print("----------------DENSE----------------")
    for idx, dense_point in enumerate(dense_points.points):
        print(
            f"{idx} {dense_point.payload['id']} {dense_point.score} {'match' if article_id == dense_point.payload['id'] else ''}"
        )

    print("----------------SPARSE----------------")
    for idx, sparse_point in enumerate(sparse_points.points):
        print(
            f"{idx} {sparse_point.payload['id']} {sparse_point.score} {'match' if article_id == sparse_point.payload['id'] else ''}"
        )

    print("----------------LATE----------------")
    for idx, late_point in enumerate(late_points.points):
        print(
            f"{idx} {late_point.payload['id']} {late_point.score} {'match' if article_id == late_point.payload['id'] else ''}"
        )


def evaluate_retrieval(documents: Iterable[dict]):
    # iterate over batch of documents wibatch_sizeuments

    results = []
    for doc in tqdm(documents.iter_rows(named=True), total=len(documents)):
        dense_points = qdrant_client.query_points(
            "hybrid-search",
            query=doc["dense_embedding"],
            using="all-MiniLM-L6-v2",
            with_payload=True,
            limit=10,
        )
        sparse_points = qdrant_client.query_points(
            "hybrid-search",
            query=models.SparseVector(**doc["sparse_embedding"].as_object()),
            using="bm25",
            with_payload=True,
            limit=10,
        )

        prefetch = [
            models.Prefetch(
                query=doc["dense_embedding"],
                using="all-MiniLM-L6-v2",
                limit=10,
            ),
            models.Prefetch(
                query=models.SparseVector(**doc["sparse_embedding"].as_object()),
                using="bm25",
                limit=10,
            ),
        ]
        late_interaction_points = qdrant_client.query_points(
            "hybrid-search",
            prefetch=prefetch,
            query=doc["late_interaction_embedding"],
            using="colbertv2.0",
            with_payload=True,
            limit=10,
        )

        result = {
            "image_index": doc["image_index"],
            "image_id": doc["image_id"],
            "article_id": doc["article_id"],
            "dense_points_chunk_index": [point.payload["index"] for point in dense_points.points],
            "dense_points_article_id": [point.payload["id"] for point in dense_points.points],
            "dense_points_score": [point.score for point in dense_points.points],
            "sparse_points_chunk_index": [
                point.payload["index"] for point in sparse_points.points
            ],
            "sparse_points_article_id": [point.payload["id"] for point in sparse_points.points],
            "sparse_points_score": [point.score for point in sparse_points.points],
            "late_points_chunk_index": [
                point.payload["index"] for point in late_interaction_points.points
            ],
            "late_points_article_id": [
                point.payload["id"] for point in late_interaction_points.points
            ],
            "late_points_score": [point.score for point in late_interaction_points.points],
        }
        results.append(result)
        # print(result)
    return pl.DataFrame(results)


with open("results_v3.pkl", "rb") as f:
    results_df = pl.DataFrame(pickle.load(f))

results_df.head()

eval = evaluate_retrieval(results_df)

eval[
    :,
    [
        "image_id",
        "article_id",
        "dense_points_article_id",
        "sparse_points_article_id",
        "late_points_article_id",
    ],
]

eval.write_json("results_v5_debug.json")


## Create collection
if not qdrant_client.collection_exists("dummy"):
    qdrant_client.create_collection(
        "dummy",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=300,
                distance=models.Distance.COSINE,
            ),
        },
    )

qdrant_client.update_vectors(
    collection_name="dummy",
    vectors_config={
        "all": models.VectorParamsDiff(
            size=300,
            distance=models.Distance.COSINE,
        ),
    },
)
