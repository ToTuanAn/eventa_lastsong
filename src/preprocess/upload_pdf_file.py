import zipfile
import os
import subprocess
from tqdm import tqdm

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def upload_to_gcs(zip_path, gcs_path):
    subprocess.run(['gsutil', 'cp', zip_path, gcs_path], check=True)

def delete_file(path):
    os.remove(path)

if __name__ == "__main__":
    doc_path = '/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder'

    for folder in tqdm(sorted(os.listdir(doc_path))):
        if int(folder) < 430:
            continue

        folder_to_zip = f'/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder/{folder}'
        zip_filename = f'/home/totuanan/Workplace/eventa_lastsong/data/Release/Document_Pdf_Folder/{folder}.zip'
        gcs_destination = f'gs://eventa_pdf_bucket/pdf_files/{folder}.zip'

        zip_folder(folder_to_zip, zip_filename)
        print(f"Finished zip {folder}")
        upload_to_gcs(zip_filename, gcs_destination)
        print(f"Finished cp {folder} to gcs")
        delete_file(zip_filename)
        print(f"Finished delete zip {folder}")