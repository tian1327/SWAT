#!/bin/bash

datasets=(
    # "semi-aves" 
    # "flowers102" 
    # "fgvc-aircraft" 
    # "eurosat" 
    # "dtd" 
    # "oxford_pets" 
    # "food101" 
    # "stanford_cars" 
    "imagenet"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "FT_mixed on $dataset"
    bash scripts/run_dataset_seed_finetune_mixed.sh $dataset 1
done



