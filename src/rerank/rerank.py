import json

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def compute_score(model, tokenizer, pairs):
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        ).to("cuda:0")

        scores = (
            model(**inputs)
            .logits.view(
                -1,
            )
            .float()
            .cpu()
            .tolist()
        )

    return scores


data_path = "../../data"

chunks_df = (
    pl.read_parquet(f"{data_path}/processed/database_v4.parquet")
    .unique(["article_id", "chunk_index"])
    .sort(["article_id", "chunk_index"])
)

documents_df = chunks_df.group_by("article_id", maintain_order=True).agg(
    content=pl.col("chunk_content").str.join("\n")
)
# model_name = "BAAI/bge-reranker-v2-m3"
model_name = "jinaai/jina-reranker-v2-base-multilingual"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)
reranker = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
).to("cuda:0")
reranker.eval()

# %%
article_list_files = [
    "colqwen3b_bm42_jina_v3_hotpot_reranked.csv",
    "jina_v2_latechunking_query_result_top_20.csv",
    "colqwen_3b_result.csv",
]


article_df = None
for article_list_file in article_list_files:
    curr_df = pl.read_csv(article_list_file)
    curr_df = curr_df.with_columns(
        article_list=pl.col("article_list").str.extract_all(r"([0-9a-f]+)"),
        article_score=pl.col("article_score")
        .str.extract_all(r"([0-9.]+)")
        .cast(pl.List(pl.Float64)),
    )

    if article_df is None:
        article_df = curr_df
    else:
        article_df = article_df.join(
            curr_df, on=["query_index", "caption"], how="left", suffix="_curr"
        )

        article_df = article_df.with_columns(
            article_list=pl.col("article_list").list.concat(pl.col("article_list_curr")),
            article_score=pl.col("article_score").list.concat(pl.col("article_score_curr")),
        ).select(
            "query_index",
            "caption",
            "article_list",
            "article_score",
        )

# %%
result = article_df
reranked_article_lists = []
reranked_article_scores = []
for row in tqdm(result.iter_rows(named=True), total=len(result)):
    query: str = row["caption"]
    article_list: list[str] = row["article_list"]
    # print(f"Reranking for query: {query}")
    # print(f"Article list: {article_list}")

    contents = []
    for article_id in article_list:
        content: str = documents_df.filter(pl.col("article_id") == article_id)[0, "content"]

        contents.append([query, content])

    # print(contents)
    if hasattr(reranker, "compute_score") and callable(getattr(reranker, "compute_score")):
        new_scores = reranker.compute_score(contents)
    else:
        new_scores = compute_score(reranker, tokenizer, contents)

    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)

    new_article_list = [article_list[i] for i, _ in ranking]
    new_article_score = [score for _, score in ranking]
    reranked_article_lists.append(new_article_list)
    reranked_article_scores.append(new_article_score)


# %%
article_lists_str = [json.dumps(article_id) for article_id in reranked_article_lists]
article_scores_str = [json.dumps(scores) for scores in reranked_article_scores]


# %%
result = result.with_columns(
    article_list=pl.Series(article_lists_str),
    article_score=pl.Series(article_scores_str),
)

result.write_csv(f"colqwen3b_bm42_jina_v3_jina_v2_late_chunking_hotpot_reranked.csv")

# %%
