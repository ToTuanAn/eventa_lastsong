from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import TripletTableSerializer
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
    BaseTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DoclingDocument,
)
from docling_core.types.doc.labels import (
    DocItemLabel,
)
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer
from typing_extensions import override


class ChunkingDocSerializer(MarkdownDocSerializer):
    """Doc serializer used for chunking purposes."""

    table_serializer: BaseTableSerializer = TripletTableSerializer()
    params: MarkdownParams = MarkdownParams(
        image_mode=ImageRefMode.PLACEHOLDER,
        # image_placeholder="",
        escape_underscores=False,
        escape_html=False,
    )


class ChunkingSerializerProvider(BaseSerializerProvider):
    """Serializer provider used for chunking purposes."""

    @override
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        """Get the associated serializer."""
        return ChunkingDocSerializer(doc=doc)


data_path = "../../data"
documents_path = f"{data_path}/raw/2/"
train_documents_path = f"{data_path}/raw/train-documents/"

document_ids = (
    pl.DataFrame(
        [
            document_id[:-4] if os.path.isfile(f"{documents_path}/{document_id}") else None
            for document_id in os.listdir(documents_path)
        ],
        schema=["article_id"],
    )
    .sort("article_id")
    .unique()
)


# train_df = pl.read_csv(f"{data_path}/raw/Train Set/gt_train.csv")
# database_df = pl.read_json(f"{data_path}/raw/database/database.json")
# database_df = (
#     database_df.transpose(include_header=True, header_name="id", column_names=["article"])
#     .unnest("article")
#     .rename({"id": "article_id"})
# )

# temp = database_df.join(document_ids, on="article_id", how="left")
# temp.null_count()

# data = database_df.join(
#     train_df["retrieved_article_id"].unique().to_frame(),
#     left_on="id",
#     right_on="retrieved_article_id",
#     how="inner",
# ).with_columns(content=pl.col("title") + "\n" + pl.col("content"))

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=1536,
    merge_peers=True,
    serializer_provider=ChunkingSerializerProvider(),
)
converter = DocumentConverter()


def chunk_document(document_id: str) -> list[tuple[str, str, list[str]]]:
    try:
        final_chunks = []
        result = converter.convert(f"{documents_path}/{document_id}.pdf")
        raw_chunks = list(chunker.chunk(result.document))

        # chunks[0].meta.doc_items[1].self_ref
        for idx, chunk in enumerate(raw_chunks):
            chunk_images = set()
            for item in chunk.meta.doc_items:
                if item.label == DocItemLabel.PICTURE:
                    chunk_images.add(item.self_ref)
            final_chunks.append(
                (document_id, idx, chunker.contextualize(chunk), list(chunk_images))
            )
    except Exception as e:
        print(f"Failed to chunk {document_id}")
        print(e)
        final_chunks = [(document_id, -1, "", [])]

    return final_chunks


tmp_chunks = list(
    chunker.chunk(
        converter.convert(f"{documents_path}/{document_ids[0, 'article_id']}.pdf").document
    )
)

data = document_ids
batch_size = 512
for idx in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size):
    data_batch = data[idx : idx + batch_size]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(chunk_document, document_id)
            for document_id in data_batch["article_id"]
        ]

    chunks = []
    for future in as_completed(futures):
        result = future.result()
        chunks.extend(result)

    chunks_df = pl.DataFrame(
        chunks,
        schema=["article_id", "chunk_index", "chunk_content", "chunk_images"],
        orient="row",
    )
    chunks_df.write_parquet(
        f"{data_path}/processed/documents/data_{data_batch['article_id'][0]}.parquet"
    )
