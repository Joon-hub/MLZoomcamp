import os
import zipfile
import kagglehub
import shutil
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Initialize Kaggle API
# api = KaggleApi()
# api.authenticate()

# Define the dataset and download path
dataset = 'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'
download_path = 'data/raw'

# Create the directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download the latest version of the dataset
print("Starting data download...")
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print(f"Downloaded to: {path}")

# Specify the dataset path
dataset_path = "/root/.cache/kagglehub/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1"

# List all files and directories in the dataset folder
for root, dirs, files in os.walk(dataset_path):
    print(f"Directory: {root}")
    for dir_name in dirs:
        print(f"  Subdirectory: {dir_name}")
    for file_name in files:
        print(f"  File: {file_name}")


# Copy the downloaded files to the 'raw' folder
print("Copying files to 'raw' folder...")
for root, dirs, files in os.walk(path):
    for file in files:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(download_path, file)
        shutil.copy2(src_path, dst_path)

print(f"Files copied to: {download_path}")
