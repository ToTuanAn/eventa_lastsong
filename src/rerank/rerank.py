from fastembed.rerank.cross_encoder import TextCrossEncoder
from transformers import AutoModelForSequenceClassification
import polars as pl
from tqdm import tqdm
import json

data_path = "../../data"

chunks_df = (
    pl.read_parquet(f"{data_path}/processed/database_v4.parquet")
    .unique(["article_id", "chunk_index"])
    .sort(["article_id", "chunk_index"])
)

documents_df = chunks_df.group_by("article_id", maintain_order=True).agg(
    content=pl.col("chunk_content").str.join("\n")
)

reranker = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
).to("cuda:0")
reranker.eval()


# %%
result = pl.read_csv("reranked_bm24_jina_v3_top_10.csv")
result = result.with_columns(
    article_list=pl.col("article_list").str.extract_all(r"([0-9a-f]+)"),
    article_score=pl.col("article_score").str.extract_all(r"([0-9.]+)").cast(pl.List(pl.Float64)),
)

colqwen_result = pl.read_csv("colqwen3b_bge_top_5.csv")
colqwen_result = colqwen_result.with_columns(
    article_list=pl.col("article_list").str.extract_all(r"([0-9a-f]+)"),
    article_score=pl.col("article_score").str.extract_all(r"([0-9.]+)").cast(pl.List(pl.Float64)),
).select("query_index", "caption", "article_list", "article_score")

result = result.join(colqwen_result, on=["query_index", "caption"], how="left", suffix="_colqwen")
result = result.with_columns(
    article_list=pl.col("article_list").list.concat(pl.col("article_list_colqwen")),
    article_score=pl.col("article_score").list.concat(pl.col("article_score_colqwen")),
)

# %%
new_article_lists = []
new_article_scores = []
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

    new_scores = reranker.compute_score(contents)
    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)

    new_article_list = [article_list[i] for i, _ in ranking]
    new_article_score = [score for _, score in ranking]
    new_article_lists.append(new_article_list)
    new_article_scores.append(new_article_score)


# %%
new_article_lists = [json.dumps(article_id) for article_id in new_article_lists]
new_article_scores = [json.dumps(scores) for scores in new_article_scores]

origin_result = pl.read_csv("colqwen3b_bge_top_5.csv")
origin_result = origin_result.with_columns(
    new_article_list=pl.Series(new_article_lists),
    new_article_score=pl.Series(new_article_scores),
)

origin_result.write_csv(f"colqwen3b_bm42_jina_v3_hotpot_reranked_{model_name}).csv")
# %%
