# %%
import argparse
from transformers import pipeline
import os

args = argparse.ArgumentParser()
args.add_argument("--data_dir", type=str, default="/Users/lap14888/Documents/eventa_lastsong/data/")
args.add_argument("--test_file", type=str, default="colqwen_trained_result.csv")
args.add_argument("--top_k", type=int, nargs="+", default=[1, 2, 3, 5])

args = args.parse_args()

data_dir = args.data_dir
test_file = args.test_file
test_file_name = os.path.splitext(test_file)[0]

summarizer = pipeline("summarization", model="zpdeaccount/bart-finetuned-pressrelease")

# %%
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
ckpt = "google/siglip2-so400m-patch16-512"
model = AutoModel.from_pretrained(ckpt).eval()
processor = AutoProcessor.from_pretrained(ckpt)


# %%
from qdrant_client import QdrantClient

url = "http://localhost:6333"
client = QdrantClient(url=url)

collection_name = "database_siglipv2"

# %%
import torch
def process_text(summarizer, text):
    summary_text = summarizer(text, do_sample=False)
    text = summary_text[0]['summary_text']
    return text


def embed_text(model, processor, text):
    with torch.no_grad():
        text_inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=64).to(model.device)
    return model.get_text_features(**text_inputs)



# %%
import pandas as pd

test_path = os.path.join(data_dir, test_file)
test_df = pd.read_csv(test_path)

# %%
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
import json
from tqdm import tqdm

top_ks = args.top_k
top_image = 100
for top_k in tqdm(top_ks):
    final_df = test_df.copy()
    image_ids_results = []
    image_ids_scores = []
    final_image_ids_results = []
    final_image_ids_scores = []
    for index, row in tqdm(test_df.iterrows()):
        # Extract the text from the row
        caption = row['caption']
        available_articles = json.loads(str(row['article_list']).replace("'", '"'))[:top_k]
        article_scores = json.loads(row['article_score'])[:top_k]
        
        try:
            text_embedding = embed_text(model, processor, caption)
        except Exception as e:
            caption = process_text(summarizer, caption)
            text_embedding = embed_text(model, processor, caption)
            
        search_result = client.query_points(
            collection_name=collection_name,
            query=text_embedding.squeeze().tolist(),
            with_payload=True,
            query_filter=Filter(
                should=[FieldCondition(key="retrieved_article_id", match=MatchAny(any=available_articles))]
            ) if available_articles else None,
            limit=top_image
        ).points
        
        image_ids = []
        image_scores = []
        final_results = []
        for point in search_result:
            image_ids.append(point.payload['retrieved_image_id'])
            image_scores.append(point.score)
            final_results.append((point.payload['retrieved_image_id'], point.score * article_scores[available_articles.index(point.payload['retrieved_article_id'])]))
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:min(10, len(final_results))]
        image_ids_results.append(json.dumps(image_ids[:min(10, len(image_ids))]))
        image_ids_scores.append(json.dumps(image_scores[:min(10, len(image_ids))]))
        final_image_ids_results.append(json.dumps([f[0] for f in final_results]))
        final_image_ids_scores.append(json.dumps([f[1] for f in final_results]))
    final_df["image_list"] = image_ids_results
    final_df["image_score"] = image_ids_scores
    final_df["final_image_list"] = final_image_ids_results
    final_df["final_image_score"] = final_image_ids_scores
    final_df.to_csv(os.path.join(data_dir, f"{test_file_name}_top_{top_k}.csv"), index=False)

# %%



