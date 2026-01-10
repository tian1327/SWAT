#!/bin/bash

# List of folders you want to loop through
folders=(
  "fgvc-aircraft"
  "flowers102"
  "food101"
  "oxford_pets"
  "semi-aves"
  "stanford_cars"
)

# Your rclone destination
remote="mytamugdrive:temp"

for folder in "${folders[@]}"; do
  echo "Copying from $folder to $remote..."
  
  rclone copy "$folder" "$remote" \
    --max-depth 1 \
    --exclude "*.pkl" \
    -P
  
  echo "Done with $folder"
done

