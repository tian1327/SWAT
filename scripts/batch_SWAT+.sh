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
    echo "ablate SWAT+ stage 2 few-shot finetuning $dataset"

    bash scripts/run_dataset_seed_SWAT+.sh $dataset

done



