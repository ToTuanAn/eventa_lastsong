import pandas as pd
import yaml
from tqdm import tqdm

from src.model.document_emb.colqwen2_5 import ColQwen25Model
from src.retrieval.colqwen2_5_indexer import Colqwen25Retrieval

if __name__ == "__main__":
    df = pd.read_csv("/home/totuanan/Workplace/eventa_lastsong/data/private_query.csv")

    with open("/home/totuanan/Workplace/eventa_lastsong/src/config/model/colqwen2_5.yaml", "r") as file:
        model_config = yaml.safe_load(file)

    model = ColQwen25Model(model_config)

    with open("/home/totuanan/Workplace/eventa_lastsong/src/config/indexer/colqwen2_5_retrieval_baseline.yaml", "r") as file:
        retrieval_config = yaml.safe_load(file)

    retrieval = Colqwen25Retrieval(retrieval_config)

    query_index_list, caption_list, retrieval_list, retrieval_score, label_list, acc_at1_list, acc_at3_list, acc_at5_list, acc_at10_list = [], [], [], [], [], [], [], [], []

    for idx, row in tqdm(df.iterrows()):
        query_index = row["query_index"]
        input = row["query_text"]

        input_embedding = model.get_query_embedding(input).cpu().float().numpy().tolist()

        # label = row["retrieved_article_id"]

        for i in range(5):
            try:
                response = retrieval.query(input_embedding)
                break
            except Exception as e:
                print(f"Error: , {e}")
                print(f"Fail this time. Retry {i} time")

        article_list = []
        article_score = []

        for i in range(10):
            article_list.append(response.points[i].payload['article_id'])
            article_score.append(response.points[i].score)

        #
        # acc_at1 = any(item == label for item in article_list[:1])
        # acc_at3 = any(item == label for item in article_list[:3])
        # acc_at5 = any(item == label for item in article_list[:5])
        # acc_at10 = any(item == label for item in article_list[:10])

        print(f"Finished {str(idx)} caption")

        if int(int(idx)+1) % 1000 == 0:
            # print("Accuracy at 1: ", sum(acc_at1_list) / len(acc_at1_list))
            # print("Accuracy at 3: ", sum(acc_at3_list) / len(acc_at3_list))
            # print("Accuracy at 5: ", sum(acc_at5_list) / len(acc_at5_list))
            # print("Accuracy at 10: ", sum(acc_at10_list) / len(acc_at10_list))
            #
            # with open("/data/evaluation_prefetch5000.txt", "a") as f:
            #     f.write(f"Evaluation for {idx+1} captions \n")
            #     f.write(f"Accuracy at 1: {sum(acc_at1_list) / len(acc_at1_list)} \n")
            #     f.write(f"Accuracy at 3: {sum(acc_at3_list) / len(acc_at3_list)} \n")
            #     f.write(f"Accuracy at 5: {sum(acc_at5_list) / len(acc_at5_list)} \n")
            #     f.write(f"Accuracy at 10: {sum(acc_at10_list) / len(acc_at10_list)} \n")

            result_df = pd.DataFrame({
                "query_index": query_index_list,
                "caption": caption_list,
                "article_list": retrieval_list,
                "article_score": retrieval_score,
                # "label": label_list,
                # "acc@1": acc_at1_list,
                # "acc@3": acc_at3_list,
                # "acc@5": acc_at5_list,
                # "acc@10": acc_at10_list
            })

            result_df.to_csv("/home/totuanan/Workplace/eventa_lastsong/data/private_colqwen_result.csv", index=False)

        caption_list.append(input)
        retrieval_list.append(article_list)
        query_index_list.append(query_index)
        retrieval_score.append(article_score)
        # label_list.append(label)
        # acc_at1_list.append(acc_at1)
        # acc_at3_list.append(acc_at3)
        # acc_at5_list.append(acc_at5)
        # acc_at10_list.append(acc_at10)

    # print("Accuracy at 1: ", sum(acc_at1_list)/len(acc_at1_list))
    # print("Accuracy at 3: ", sum(acc_at3_list) / len(acc_at3_list))
    # print("Accuracy at 5: ", sum(acc_at5_list) / len(acc_at5_list))
    # print("Accuracy at 10: ", sum(acc_at10_list) / len(acc_at10_list))

    result_df = pd.DataFrame({
        "query_index": query_index_list,
        "caption": caption_list,
        "article_list": retrieval_list,
        "article_score": retrieval_score,
        # "label": label_list,
        # "acc@1": acc_at1_list,
        # "acc@3": acc_at3_list,
        # "acc@5": acc_at5_list,
        # "acc@10": acc_at10_list
    })

    result_df.to_csv("/home/totuanan/Workplace/eventa_lastsong/data/private_colqwen_result.csv", index=False)