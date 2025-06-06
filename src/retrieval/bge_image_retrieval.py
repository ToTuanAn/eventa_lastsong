from qdrant_client import QdrantClient, models
from src.model.document_emb.colqwen2_5 import ColQwen25Model
from tqdm import tqdm
import os
import yaml
from pdf2image import convert_from_path
import glob
import json
import numpy as np
import pandas as pd 
import ast
from src.model.image_emb.bge import BgeVisualizedModel

TOP_K_ARTICLE_LIST = [8, 9, 10]

if __name__ == "__main__":
    for k in TOP_K_ARTICLE_LIST:
        with open("/home/totuanan/Workplace/eventa_lastsong/data/database.json", "r") as f:
            database = json.load(f)

        # Load from file
        with open("/home/totuanan/Workplace/eventa_lastsong/src/config/model/bge_visualized.yaml", "r") as file:
            model_config = yaml.safe_load(file)

        model = BgeVisualizedModel(model_config)

        image_path = "/home/totuanan/Workplace/eventa_lastsong/data/database_compressed_images/database_images_compressed90"

        df = pd.read_csv("/home/totuanan/Workplace/eventa_lastsong/data/colqwen3b_bm42_jina_v3_hotpot_reranked.csv")

        df_image_list = []
        df_image_score = []
        df_final_image_list = []
        df_final_image_score = []

        for ix, row in df.iterrows():
            article_list = row["new_article_list"]
            article_score = row["new_article_score"]
            image_list_dict = {}

            result_list_dict = {
                "image_list": [],
                "image_score": [],
                "final_image_score": []
            }

            text_emb = model.embed(text=row["caption"],image_file=None)

            top_k_article_list = ast.literal_eval(article_list)[:k]
            top_k_article_score = ast.literal_eval(article_score)[:k]

            article_dict = dict(zip(top_k_article_list, top_k_article_score))

            for article in top_k_article_list:
                print(article)
                image_list_dict[article] = {}
                image_list_dict[article]["images"] =  database[article]["images"]
            
            for article in image_list_dict:
                
                for idx in range(len(image_list_dict[article]["images"])):

                    image_id = image_list_dict[article]["images"][idx]
                    image_file = f"{image_path}/{image_id}.jpg"

                    if os.path.exists(image_file):
                        # if len(database[article]["content"]) >= 29000:
                        #     database[article]["content"] = database[article]["content"][:29000]

                        image_emb = model.embed(image_file=image_file,text=None)
                        

                        if image_emb is not None:
                            score = (text_emb @ image_emb.T).cpu().item()
                            result_list_dict["image_list"].append(image_id) 
                            result_list_dict["image_score"].append(score)
                            result_list_dict["final_image_score"].append(score*article_dict[article])

                    else:
                        print(f"Missing pdf: {image_file}")

            
            sorted_image_list_pairs = sorted(zip(result_list_dict["image_score"], result_list_dict["image_list"]), reverse=True)
            sorted_final_image_list_pairs = sorted(zip(result_list_dict["final_image_score"], result_list_dict["image_list"]), reverse=True)
            
            if len(sorted_image_list_pairs) == 0:
                image_score, image_list = [], []
                final_image_score, final_image_list = [], []
            else:
                image_score, image_list = zip(*sorted_image_list_pairs)
                final_image_score, final_image_list = zip(*sorted_final_image_list_pairs)

            print("Image score: ", image_score)
            print("Image list: ", image_list)
            print("Final image score", final_image_score)
            print("Final image list", final_image_list)

            if len(sorted_image_list_pairs) == 0:
                df_image_score.append(image_score)
                df_image_list.append(image_list)
                df_final_image_score.append(final_image_score)
                df_final_image_list.append(final_image_list)
            else:
                df_image_score.append(list(image_score))
                df_image_list.append(list(image_list))
                df_final_image_score.append(list(final_image_score))
                df_final_image_list.append(list(final_image_list))

            print(f'Finished {ix}')
        
        df["image_list"] = df_image_list
        df["image_score"] = df_image_score
        df["final_image_list"] = df_final_image_list
        df["final_image_score"] = df_final_image_score

        df.to_csv(f"/home/totuanan/Workplace/eventa_lastsong/data/colqwen3b_bm42_jina_v3_hotpot_reranked_{k}.csv",index=False)
