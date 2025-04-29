import os
from typing import Iterable
import uuid

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
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


def get_embeddings(documents: Iterable[str]):
    dense_embeddings = list(dense_embedding_model.embed(documents))
    bm25_embeddings = list(bm25_embedding_model.embed(documents))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(documents))

    return dense_embeddings, bm25_embeddings, late_interaction_embeddings


## chunking
def upsert_data(documents: Iterable[dict], batch_size: int = 256):
    # iterate over batch of documents wibatch_sizeuments
    for idx in range(0, len(documents), batch_size):
        batch_documents = documents[idx : idx + batch_size]
        dense_embeddings, bm25_embeddings, late_interaction_embeddings = get_embeddings(
            batch_documents["content"]
        )

        points = []
        for dense_embedding, bm25_embedding, late_interaction_embedding, doc in zip(
            dense_embeddings,
            bm25_embeddings,
            late_interaction_embeddings,
            batch_documents.iter_rows(named=True),
        ):
            point = PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['id']}_chunk_{doc['index']}")),
                vector={
                    "all-MiniLM-L6-v2": dense_embedding,
                    "bm25": bm25_embedding.as_object(),
                    "colbertv2.0": late_interaction_embedding,
                },
                payload=doc,
            )
            points.append(point)

        operation_info = qdrant_client.upsert(collection_name="hybrid-search", points=points)
        print(operation_info)


data = pl.read_json("./train_article_chunked_v3_more_info.json")
upsert_data(data[5750:], batch_size=16)
