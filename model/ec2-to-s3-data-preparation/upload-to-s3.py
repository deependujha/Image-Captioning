import boto3
import os
from os import walk

s3 = boto3.client("s3")

image_folder = "flickr30k_images"

file_name = "results.csv"
destination = "datasets"

for idx, (dirpath, dirnames, filenames) in enumerate(walk(image_folder)):
    for idy, file_name in enumerate(filenames):
        print("\n" + "-" * 70)
        print(f"{idy=}")
        file_name = os.path.join(image_folder, file_name)
        s3_path = os.path.join(destination, file_name)
        print(f"uploading to s3 path: {s3_path}")
        print(f"uploading file: {file_name}")
        s3.upload_file(file_name, "deependu-my-personal-projects", s3_path)
print("done")
