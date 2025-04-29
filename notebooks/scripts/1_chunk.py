from io import BytesIO
import os
from typing import Iterable

from docling.chunking import HybridChunker
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
import polars as pl
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)

## Embedding
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")  # 20 M
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")  # rule based
late_interaction_embedding_model = LateInteractionTextEmbedding(  # 100 M
    "colbert-ir/colbertv2.0"
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
converter = DocumentConverter()

database_df = pl.read_json("data/database/database.json")
database_df = database_df.transpose(
    include_header=True, header_name="id", column_names=["article"]
).unnest("article")
train_df = pl.read_csv("data/Train Set/gt_train.csv")

data = database_df.join(
    train_df["retrieved_article_id"].unique().to_frame(),
    left_on="id",
    right_on="retrieved_article_id",
    how="inner",
)

del train_df
del database_df

data = data.with_columns(content=pl.col("title") + "\n" + pl.col("content"))


def chunk_document(document: str):
    document_stream = DocumentStream(name="document.md", stream=BytesIO(document.encode("utf-8")))
    results = converter.convert(document_stream)
    chunks = chunker.chunk(results.document)
    return [chunker.contextualize(chunk) for chunk in chunks]


data = data.with_columns(
    chunks=pl.col("content").map_elements(chunk_document, return_dtype=pl.List(pl.Utf8))
)

# data = pl.read_json("./train_article_chunked_v1.json")

data = (
    data.explode("chunks")
    .with_columns(
        chunk_index=pl.col("id").rank("ordinal").over("id"),
        content=pl.col("chunks"),
        article_id=pl.col("id"),
    )
    .select(["article_id", "url", "date", "title", "images", "content", "chunk_index"])
    .sort("article_id", "chunk_index")
)

original_content_df = data.group_by("article_id", maintain_order=True).agg(
    original_content=pl.col("content").str.join("")
)

data = data.join(original_content_df, on="article_id", how="left").sort(
    "article_id", "chunk_index"
)

data = data.with_columns(chunk_length=pl.col("content").str.len_chars()).with_columns(
    chunk_start_index=pl.col("chunk_length").cum_sum().shift(fill_value=0).over("article_id")
)

data = data.with_columns(
    sliced_content=pl.col("original_content").str.slice(
        pl.col("chunk_start_index"), pl.col("chunk_length")
    )
)

# is content equal to slice_content
is_content_equal_df = data["content"] == data["sliced_content"]
is_content_equal_df.sum()

data.write_json("./data/processed/train_article_chunked_v5_more_info.json")


def get_embeddings(documents: Iterable[str]):
    dense_embeddings = list(dense_embedding_model.embed(documents))
    bm25_embeddings = list(bm25_embedding_model.embed(documents))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(documents))

    return dense_embeddings, bm25_embeddings, late_interaction_embeddings


dense_embeddings, bm25_embeddings, late_interaction_embeddings = get_embeddings(
    data["content"][:1]
)

## Create collection
if not qdrant_client.collection_exists("hybrid-search"):
    qdrant_client.create_collection(
        "hybrid-search",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,  # Disable HNSW graph creation
                ),
            ),
        },
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
