import zipfile
import os
from tqdm import tqdm
import glob
import json

def gcs():

    # Path to your zip file
    doc_path = '/home/totuanan/Workplace/eventa_lastsong/data/pdf_files'


    for zip_file in range(1, 438):
        zip_file_name = str(zip_file)
        if os.path.exists(f"{doc_path}/{zip_file_name}.zip"):
            continue

        try:
            os.system(f"gsutil cp gs://eventa_pdf_bucket/pdf_files/{zip_file_name}.zip /home/totuanan/Workplace/eventa_lastsong/data/pdf_files/{zip_file_name}.zip")
        except:
            continue
    print("Extraction complete!")

def extract_gcs():
    import zipfile
    import os
    import shutil

    doc_path = '/home/totuanan/Workplace/eventa_lastsong/data/pdf_files'
    
    with open("/home/totuanan/Workplace/eventa_lastsong/data/database.json", "r") as f:
        database = json.load(f)

    print(len(database))

    print(len(glob.glob(f"{doc_path}/*.pdf")))

    # pdf_files = [zip_file.split("/")[-1].replace(".pdf", "") for zip_file in glob.glob(f"{doc_path}/*.pdf")]
    # print(pdf_files)

    # with open("/home/totuanan/Workplace/eventa_lastsong/data/missing_id.csv", "a") as f:
    #     for key in database:
    #         if key not in pdf_files:
    #             f.write(key+"\n")
            
    

    for zip_file in tqdm(glob.glob(f"{doc_path}/437/*.pdf")):
        zip_file_name = zip_file.split("/")[-1]
        # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #     zip_ref.extractall(doc_path)

        # if zip_file_name[-3:] != "pdf":
        #     print(zip_file_name)

        # if os.path.exists(zip_file):
        #     os.rename(zip_file,f"{doc_path}/{zip_file_name}")
        # os.remove(f"{doc_path}/{zip_file_name}")
    # shutil.rmtree(f"{doc_path}/437")
    print("Extraction complete!")
    
extract_gcs()