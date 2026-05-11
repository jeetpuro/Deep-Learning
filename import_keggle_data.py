# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("fragdude44/dataset")

# print("Path to dataset files:", path)



import shutil
import kagglehub

path = kagglehub.dataset_download("fragdude44/dataset")

target_folder = "/home/admin1/Delete_when_done/data"

shutil.copytree(path, target_folder, dirs_exist_ok=True)

print("Copied to:", target_folder)