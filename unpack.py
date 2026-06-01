import zipfile
import os

with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

def extract_nested_zips(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_folder = zip_path.replace(".zip", "")
                print(f"Unpacked: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

extract_nested_zips("data")