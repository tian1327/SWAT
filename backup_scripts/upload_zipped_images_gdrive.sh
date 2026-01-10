#!/bin/bash

# Directory containing the zipped files
zipped_dir="/scratch/group/real-fs/retrieved/zipped_images"

# Your rclone destination
remote="mytamugdrive:temp"

echo "Uploading all zip files from $zipped_dir to $remote..."

rclone copy "$zipped_dir" "$remote" \
  --include "*.zip" \
  -P

echo "Done uploading all zipped files"

