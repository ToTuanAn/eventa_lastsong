import os
import pandas as pd
import glob
import random
import json
from PIL import Image
import io

from pdf2image import convert_from_path


def data_cleaning(training_csv_path="/home/totuanan/Workplace/eventa_lastsong/data/gt_train.csv"):
    df = pd.read_csv(training_csv_path)

    pdf_folder = "/home/totuanan/Workplace/eventa_lastsong/data/pdf_files"

    idx = 1
    training_json = []
    total_records = len(df)

    for idx, row in df.iterrows():
        article_id = row["retrieved_article_id"]
        caption = row["caption"]

        file = f"{pdf_folder}/{article_id}.pdf"
        images = convert_from_path(file, dpi=300)

        print("Num pages: ", len(images))
        for j, image in enumerate(images):

            if len(training_json) >= total_records // 5:
                random.shuffle(training_json)
                with open(f"/home/totuanan/Workplace/eventa_lastsong/data/training_json_2/training_{idx}.json", "w", encoding="utf-8") as f:
                    json.dump(training_json, f, indent=2, ensure_ascii=False)
                os.system(f"gsutil cp /home/totuanan/Workplace/eventa_lastsong/data/training_json_2/training_{idx}.json gs://eventa_pdf_bucket/training_json_2/training_{idx}.json")
                training_json = []

            training_json.append({
                "image": file,
                "page": j,
                "article_id": article_id,
                "query": caption
            })

        print(f"Finished caption {idx}")

    with open(f"/home/totuanan/Workplace/eventa_lastsong/data/training_json_2/training_{idx}.json", "w", encoding="utf-8") as f:
        json.dump(training_json, f, indent=2, ensure_ascii=False)
    os.system(f"gsutil cp /home/totuanan/Workplace/eventa_lastsong/data/training_json_2/training_{idx}.json gs://eventa_pdf_bucket/training_json_2/training_{idx}.json")


if __name__ == "__main__":
    data_cleaning()