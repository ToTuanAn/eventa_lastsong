import uuid
from qdrant_client import QdrantClient, models
from src.model.document_emb.colqwen2_5 import ColQwen25Model
from tqdm import tqdm
import yaml
from pdf2image import convert_from_path
import glob
import json
import numpy as np

class Colqwen25Retrieval:
    def __init__(self, config):
        print(config)
        self.client = QdrantClient(url=config["qdrant"]['url'],
                                    api_key=config["qdrant"]['api_key'], timeout=3600)

        self.collection_name = config["qdrant"]["collection_name"]
        self.prefetch_limit = config["params"]["prefetch_limit"]
        self.search_limit = config["params"]["search_limit"]

    def create(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "original":
                        models.VectorParams(  # switch off HNSW
                            size=128,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                            hnsw_config=models.HnswConfigDiff(
                                m=0  # switching off HNSW
                            )
                        ),
                    "mean_pooling_columns": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                    "mean_pooling_rows": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )
        except Exception as e:
            print("error: ", e)


    def upload_batch(self, original_batch, pooled_by_rows_batch, pooled_by_columns_batch, payload_batch, collection_name):
        self.client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch
            },
            payload=payload_batch
        )


    def query(self, query_embedding):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    limit=self.prefetch_limit,
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query_embedding,
                    limit=self.prefetch_limit,
                    using="mean_pooling_rows"
                ),
            ],
            limit=self.search_limit,
            with_payload=True,
            using="original"
        )

        return response

if __name__ == '__main__':
    batch_size = 1  # based on available compute

    with open("/home/totuanan/Workplace/eventa_lastsong/data/database.json", "r") as f:
        database = json.load(f)

    # Load from file
    with open("/home/totuanan/Workplace/eventa_lastsong/src/config/model/colqwen2_5.yaml", "r") as file:
        model_config = yaml.safe_load(file)

    model = ColQwen25Model(model_config)

    with open("/home/totuanan/Workplace/eventa_lastsong/src/config/indexer/colqwen2_5_retrieval.yaml", "r") as file:
        retrieval_config = yaml.safe_load(file)


    retrieval = Colqwen25Retrieval(retrieval_config)
    try:
        response = retrieval.client.get_collection(retrieval.collection_name)
    except:
        retrieval.create()

    idx = 0

    for pdf_file in tqdm(glob.glob(f"/home/totuanan/Workplace/eventa_lastsong/data/temp/*.pdf")):
        image_batch = convert_from_path(pdf_file, dpi=300)
        article_id = pdf_file.split("/")[-1].replace(".pdf", "")

        original_batch, pooled_by_rows_batch, pooled_by_columns_batch = [], [], []
        
        for idx in range(len(image_batch)):
            image = [image_batch[idx]]

            original_image, pooled_by_rows_image, pooled_by_columns_image = model.embed_and_mean_pool_batch(image)
            original_batch.append(original_image[0])
            pooled_by_rows_batch.append(pooled_by_rows_image[0])
            pooled_by_columns_batch.append(pooled_by_columns_image[0])

        original_batch_flatten, pooled_by_rows_batch_flatten, pooled_by_columns_batch_flatten = [], [], []

        for idx in range(len(original_batch)):
            original_batch_flatten.extend(original_batch[idx])

        for idx in range(len(pooled_by_rows_batch)):
            pooled_by_rows_batch_flatten.extend(pooled_by_rows_batch[idx])

        for idx in range(len(pooled_by_columns_batch)):
            pooled_by_columns_batch_flatten.extend(pooled_by_columns_batch[idx])

        if len(original_batch_flatten) * 128 >= 1048576:
            print("Exceed Vector Size")
            original_batch_flatten = original_batch_flatten[:8190]

        print("Page Lens: ", len(original_batch), len(original_batch[0]), len(original_batch_flatten))
        print("Pooled By Rows Dim: ", len(pooled_by_rows_batch), len(pooled_by_rows_batch[0]), len(pooled_by_rows_batch_flatten))
        print("Pooled By Cols Dim: ", len(pooled_by_columns_batch), len(pooled_by_columns_batch[0]), len(pooled_by_columns_batch_flatten))

        metadata = database[article_id]
        upsert_metadata = {"images": metadata["images"]}

        try:
            retrieval.upload_batch(
                            np.asarray([original_batch_flatten], dtype=np.float32),
                            np.asarray([pooled_by_rows_batch_flatten], dtype=np.float32),
                            np.asarray([pooled_by_columns_batch_flatten], dtype=np.float32),
                            [
                                {
                                    "article_id": article_id,
                                    "metadata": upsert_metadata
                                }
                            ],
                            retrieval.collection_name
                        )
        except Exception as e:
            with open("/home/totuanan/Workplace/eventa_lastsong/fail.txt", "a") as f:
                f.write(article_id)
                f.write("\n")
            print(f"Error during upsert: {e}")
            continue


    # with tqdm(total=len(dataset),
    #           desc=f"Uploading progress of \"{dataset_source}\" dataset to \"{collection_name}\" collection") as pbar:
    #     for i in range(0, len(dataset), batch_size):
    #         batch = dataset[i: i + batch_size]
    #         image_batch = batch["image"]
    #         current_batch_size = len(image_batch)
    #         # Update the progress bar
    #         original_batch, pooled_by_rows_batch, pooled_by_columns_batch = model.embed_and_mean_pool_batch(image_batch)
    #
    #         try:
    #             pass
    #         except Exception as e:
    #             print(f"Error during embed: {e}")
    #             continue
    #
    #         pbar.update(current_batch_size)
    # print("Uploading complete!")
