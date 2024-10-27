#!/bin/bash

datasets=(
    # "semi-aves" 
    # "flowers102" 
    # "fgvc-aircraft" 
    # "eurosat" 
    # "dtd" 
    # "oxford_pets" 
    "food101" 
    "stanford_cars" 
    # "imagenet"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "SWAT vitb16 on $dataset"
    # bash scripts/run_dataset_seed_SWAT.sh $dataset 1
    bash scripts/run_dataset_seed_SWAT_vitb16.sh $dataset 1

done



