# loop through each subfolder x in the data/ directory
# remove all folders named "pre_extracted" or "prompts" in each subfolder x

import os
import shutil
import sys

data_dir = "data"
if not os.path.exists(data_dir):
    print(f"Directory '{data_dir}' does not exist.")
    sys.exit(1)
for subfolder in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for root, dirs, files in os.walk(subfolder_path):
            for dir_name in dirs:
                if dir_name in ["pre_extracted", "prompts"]:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Removed directory: {dir_path}")
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {e}")

print("Cleanup complete.")