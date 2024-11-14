#!/bin/bash

datasets=("semi-aves" "fgvc-aircraft" "eurosat" "dtd" "flowers102" "oxford_pets" "food101" "stanford_cars" "imagenet")

# num_samples=(100 300 500 1000 2000)
num_samples=(10)


sampling_method="T2T-rank"

prompt_name="alternates"

for dataset in "${datasets[@]}"; do
    for num_sample in "${num_samples[@]}"; do

        prefix="T2T"        
        # update prefix to include the number of samples
        prefix="${prefix}${num_sample}"

        echo "Sampling for $dataset $prefix $num_sample $sampling_method $prompt_name"

        # execute python script
        python sample_retrieval.py --prefix "$prefix" --num_samples "$num_sample" \
        --sampling_method "$sampling_method" --dataset "$dataset" --prompt_name "$prompt_name"

        echo ""
    done
done


