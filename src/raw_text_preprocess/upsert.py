import pickle
import uuid

from fastembed import SparseTextEmbedding
from late_chunking import late_chunking
import polars as pl
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, PointVectors, SparseVector
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""

qdrant_client = QdrantClient(
    url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True, timeout=1000
)

collection_name = "news"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name,
        vectors_config={
            "jinaai/jina-embeddings-v2-base-en": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
            "jinaai/jina-embeddings-v3": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            ),
            "jinaai/jina-colbert-v2": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=0,  # Disable HNSW graph creation
                ),
            ),
        },
        sparse_vectors_config={
            "Qdrant/bm42-all-minilm-l6-v2-attentions": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            ),
        },
    )


jina_embed_model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3", trust_remote_code=True, device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")

bm42_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

data_path = "../../data"

chunks_df = (
    pl.read_parquet(f"{data_path}/processed/database_v4.parquet")
    .unique(["article_id", "chunk_index"])
    .sort(["article_id", "chunk_index"])
)
# image_placeholder = ("\n<!-- image -->\n", "<!-- image -->")
# chunks_df = chunks_df.with_columns(
#     pl.col("chunk_content").str.replace_all(image_placeholder[0], "")
# ).with_columns(pl.col("chunk_content").str.replace_all(image_placeholder[1], ""))

database_df = (
    pl.read_json(f"{data_path}/raw/database/database.json")
    .transpose(include_header=True, header_name="id", column_names=["article"])
    .unnest("article")
    .rename({"id": "article_id"})
    .select(pl.exclude("content"))
)

chunks_df = chunks_df.join(database_df, on="article_id", how="left").sort(
    ["article_id", "chunk_index"]
)

# data = chunks_df
# batch_size = 32
# bm42_embeddings = []
# for idx in tqdm(range(0, len(data), batch_size)):
#     batch = data[idx : idx + batch_size]
#     embeddings = list(bm42_model.embed(batch["chunk_content"].to_list()))
#     bm42_embeddings.extend(embeddings)


# with open("bm42_embeddings.pkl", "wb") as f:
#     pickle.dump(bm42_embeddings, f)

# load pickle file
with open(f"{data_path}/processed/bm42_embeddings.pkl", "rb") as f:
    bm42_embeddings = pickle.load(f)

# documents_df = chunks_df.group_by("article_id", maintain_order=True).agg(
#     pl.col("chunk_index"), pl.col("chunk_content")
# )

# chunked_corpus = documents_df["chunk_content"].to_list()

# jina_embeds = late_chunking(
#     jina_embed_model,
#     tokenizer,
#     chunked_corpus,
#     batch_size=1,
#     truncate_max_length=None,
#     long_late_chunking_embed_size=8192,
#     long_late_chunking_overlap_size=512,
# )

# jina_embeds = [embed for sublist in jina_embeds for embed in sublist]

# with open("jina_embeddings_v1.pkl", "wb") as f:
#     pickle.dump(jina_embeds, f)

with open(f"{data_path}/processed/jina_embeddings_v1.pkl", "rb") as f:
    jina_embeds = pickle.load(f)


# points = []
# for chunk, bm42_embed, jina_embed in zip(
#     chunks_df.iter_rows(named=True), bm42_embeddings, jina_embeds
# ):
#     point = PointStruct(
#         id=chunk["chunk_id"],
#         vector={
#             "jinaai/jina-embeddings-v3": jina_embed,
#             "Qdrant/bm42-all-minilm-l6-v2-attentions": bm42_embed.as_object(),
#         },
#         payload={
#             "article_id": chunk["article_id"],
#             "chunk_id": chunk["chunk_id"],
#             "chunk_index": chunk["chunk_index"],
#             "url": chunk["url"],
#             "date": chunk["date"],
#             "title": chunk["title"],
#             "images": chunk["chunk_images"],
#             "chunk_content": chunk["chunk_content"],
#             "chunk_images": chunk["chunk_images"],
#         },
#     )
#     points.append(point)

batch_size = 512
total = []
data = chunks_df[-2:]
for i in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size):
    batch = []
    for i in range(i, i + batch_size):
        if i >= len(data):
            break
        point = PointStruct(
            id=data[i, "chunk_id"],
            vector={
                "jinaai/jina-embeddings-v3": jina_embeds[i],
                "Qdrant/bm42-all-minilm-l6-v2-attentions": bm42_embeddings[
                    i
                ].as_object(),
            },
            payload={
                "article_id": data[i, "article_id"],
                "chunk_id": data[i, "chunk_id"],
                "chunk_index": data[i, "chunk_index"],
                "url": data[i, "url"],
                "date": data[i, "date"],
                "title": data[i, "title"],
                "images": data[i, "images"].to_list(),
                "chunk_content": data[i, "chunk_content"],
                "chunk_images": data[i, "chunk_images"].to_list(),
            },
        )
        batch.append(point)

    qdrant_client.upsert(collection_name=collection_name, points=batch)
