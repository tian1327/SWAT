#!/bin/bash

# List of paths to the folders we will zip
folders=(
  "/scratch/group/real-fs/retrieved/dtd/dtd_retrieved_LAION400M-all_synonyms-random"
  "/scratch/group/real-fs/retrieved/eurosat/eurosat_retrieved_LAION400M-all_synonyms-all"
  "/scratch/group/real-fs/retrieved/fgvc-aircraft/fgvc-aircraft_retrieved_LAION400M-all_synonyms-all"
  "/scratch/group/real-fs/retrieved/flowers102/flowers102_retrieved_LAION400M-all_synonyms-random"
  "/scratch/group/real-fs/retrieved/food101/food101_retrieved_LAION400M-all_synonyms-random"
  "/scratch/group/real-fs/retrieved/oxford_pets/oxford_pets_retrieved_LAION400M-all_synonyms-random"
  "/scratch/group/real-fs/retrieved/semi-aves/semi-aves_retrieved_LAION400M-all_synonyms-all"
  "/scratch/group/real-fs/retrieved/stanford_cars/stanford_cars_retrieved_LAION400M-all_synonyms-random"
)

# Loop through each folder and create a zip file, and save it to /scratch/group/real-fs/retrieved/zipped_images
for folder in "${folders[@]}"; do
  folder_name=$(basename "$folder")
  parent_dir=$(dirname "$folder")
  zip_file="/scratch/group/real-fs/retrieved/zipped_images/${folder_name}.zip"
  echo ""
  echo "Zipping $folder to $zip_file"
  echo "folder_name: $folder_name"
  echo "parent_dir: $parent_dir"
  echo "zip_file: $zip_file"

  (cd "$parent_dir" && zip -r "$zip_file" "$folder_name")
done