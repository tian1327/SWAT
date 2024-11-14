#!/bin/bash

datasets=(
    "semi-aves" 
    "flowers102" 
    "fgvc-aircraft" 
    "eurosat" 
    "dtd" 
    "oxford_pets" 
    "food101" 
    "stanford_cars" 
    "imagenet"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "ablate SWAT stage 2 no-overfit $dataset"

    bash scripts/run_dataset_seed_SWAT_ablate_stage2_epochs.sh $dataset

done



