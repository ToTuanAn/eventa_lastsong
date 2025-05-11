import os
import cv2
from tqdm import tqdm
import glob
from pdf2image import convert_from_path
import numpy as np
import shutil

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def is_image_contained(big_image, small_image, threshold):
    # Read both images
    big_image = cv2.cvtColor(np.array(big_image), cv2.COLOR_BGR2GRAY)
    small_image = cv2.cvtColor(np.array(small_image), cv2.COLOR_BGR2GRAY)

    # Template matching
    err = mse(big_image, small_image)

    # If match is above threshold, it's considered found
    if err <= threshold:
        print(f"Max similarity score: {err:.4f}")
        print("Small image is found in the big image!")
        return True
    else:
        # print("Small image is NOT found in the big image.")
        return False

def clean_pdf(DB_PATH, ADS_PATH, DELETE_PATH):
    ads_images = [convert_from_path(ads_image_path, dpi=300)[0] for ads_image_path in glob.glob(f"{ADS_PATH}/*.pdf")]

    for folder in tqdm(sorted(os.listdir(DB_PATH))):
        folder_path = os.path.join(DB_PATH, folder)
        delete_folder_path = os.path.join(DELETE_PATH, folder)

        if not os.path.exists(delete_folder_path):
            os.makedirs(delete_folder_path, exist_ok=True)

        for pdf_file in tqdm(glob.glob(f"{folder_path}/*.pdf")):
            pdf_file_name = pdf_file.split("/")[-1]

            ## PDF IMAGES
            images = convert_from_path(pdf_file, dpi=300)
            first_page = images[0]

            ## ADS IMAGES
            for first_page_ads_image in ads_images:

                if is_image_contained(first_page, first_page_ads_image, threshold=500):
                    os.rename(f"{pdf_file}", f"{delete_folder_path}/{pdf_file_name}")
                    break

def count_pdf_files(doc_path="/home/totuanan/Workplace/eventa_lastsong/data/pdf_files"):
    doc_set = set()
    my_doc = []

    for pdf_file in tqdm(glob.glob(f"{doc_path}/*.pdf")):
        pdf_file_name = pdf_file.split("/")[-1]
        if pdf_file_name not in doc_set:
            doc_set.add(pdf_file_name)
        else:
            print(pdf_file_name)
        my_doc.append(pdf_file_name)

    print("Total documents: ", len(doc_set))
    print(len(my_doc))

    return doc_set

def extract_zipfile():
    import zipfile
    import os

    # Path to your zip file
    doc_path = '/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder'

    for folder in tqdm(sorted(os.listdir(doc_path))):

        folder_path = os.path.join(doc_path, folder)
        for zip_file in tqdm(glob.glob(f"{folder_path}/*.zip")):
            zip_file_name = zip_file.split("/")[-1]

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(folder_path)

            if os.path.exists(zip_file):
                os.remove(zip_file)

            for pdf_file in tqdm(glob.glob(f"{folder_path}/*/*/*/*.pdf")):
                pdf_file_name = pdf_file.split("/")[-1]
                os.rename(f"{pdf_file}", f"{folder_path}/{pdf_file_name}")

            if os.path.exists(os.path.join(folder_path,"kaggle")):
                shutil.rmtree(os.path.join(folder_path,"kaggle"))

    print("Extraction complete!")

if __name__ == "__main__":
    count_pdf_files()
    # clean_pdf(DB_PATH="/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder",
    #           ADS_PATH="/home/totuanan/Workplace/eventa_lastsong/data/Release/Ads",
    #           DELETE_PATH="/home/totuanan/Workplace/eventa_lastsong/data/Release/Delete_Pdf_Folder")