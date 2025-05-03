import base64

import pandas as pd
import glob
import random
import json
from PIL import Image
import io

from pdf2image import convert_from_path


def data_cleaning(training_csv_path="/home/totuanan/Workplace/eventa_lastsong/data/Release/Train_Set/gt_train.csv"):
    df = pd.read_csv(training_csv_path)

    pdf_folder = "/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder"

    idx = 1
    training_json = []
    total_records = len(df)


    for idx, row in df.iterrows():
        article_id = row["retrieved_article_id"]
        caption = row["caption"]

        for file in glob.glob(f"{pdf_folder}/*/{article_id}.pdf"):
            images = convert_from_path(file, dpi=300)

            print("Num pages: ", len(images))
            for image in images:

                if len(training_json) >= total_records // 10:
                    random.shuffle(training_json)
                    with open(f"/home/totuanan/Workplace/eventa_lastsong/data/Release/Train_Set/training_json/training_{idx}.json", "w", encoding="utf-8") as f:
                        json.dump(training_json, f, indent=2, ensure_ascii=False)
                    idx += 1
                    training_json = []

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()

                training_json.append({
                    "image": base64.b64encode(img_bytes).decode('utf-8'),
                    "article_id": article_id,
                    "query": caption
                })
        break

    with open(f"/home/totuanan/Workplace/eventa_lastsong/data/Release/Train_Set/training_json/training_{idx}.json", "w", encoding="utf-8") as f:
        json.dump(training_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data_cleaning()