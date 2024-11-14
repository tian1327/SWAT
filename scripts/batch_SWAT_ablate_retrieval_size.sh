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
    echo "ablate SWAT retrieval size T2T1000 $dataset"

    bash scripts/run_dataset_seed_SWAT_ablate_retrieval_size_T2T1000.sh $dataset 1

done



