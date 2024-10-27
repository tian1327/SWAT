#!/bin/bash

datasets=("semi-aves" "fgvc-aircraft" "eurosat" "dtd" "flowers102" "oxford_pets" "food101" "stanford_cars" "imagenet")

prefix="T2T500"

num_samples=500

sampling_method="T2T-rank"

prompt_name='alternates'

for dataset in "${datasets[@]}"; do
    echo "Sampling for $dataset $prefix $num_samples $sampling_method $prompt_name"

    # execute python script
    python sample_retrieval.py --prefix "$prefix" --num_samples "$num_samples" \
    --sampling_method "$sampling_method" --dataset "$dataset" --prompt_name "$prompt_name"

    echo ""
done


