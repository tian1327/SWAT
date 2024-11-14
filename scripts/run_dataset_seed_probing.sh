#!/bin/bash

methods=("probing")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved")
data_sources=("fewshot")

# cls_inits=("random" "text" "REAL-Prompt" )
cls_inits=("REAL-Prompt")

# shot_values=(4 8 16)
shot_values=(16)

retrieval_splits=("T2T500")

batch_size=32

epochs=10

model_cfg="vitb32_openclip_laion400m"
# model_cfg="vitb16_openclip_laion400m"

log_mode="both"


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

# Check if command-line arguments were provided
if [ "$#" -ge 1 ]; then
    datasets=("$1")  # Use the provided command-line argument for the dataset
else
    echo "Usage: $0 <dataset> [seed]"
fi

seeds=(1)

# take the first item before underscore in model_cfg as the model architecture
model_arch=$(echo $model_cfg | cut -d'_' -f 1)

# combine the method and model_arch to form the folder name
folder="${methods[0]}_${model_arch}"


# Check if the results folder exists, if not create it
if [ ! -d "results/$folder" ]; then
    mkdir -p "results/$folder"
fi

output_folder="output/$folder"
if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi


# Dynamically set the filename based on the dataset
output_file="results/${folder}/${datasets[0]}.csv"


# Create or clear the output file
echo "Dataset,Method,DataSource,Init,Shots,Seed,Retrieve,Stage1Acc,Stage1_WSFT,Stage2Acc" > "$output_file"

# Loop through all combinations and run the script
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for data_source in "${data_sources[@]}"; do
            for shots in "${shot_values[@]}"; do
                for init in "${cls_inits[@]}"; do                
                    for seed in "${seeds[@]}"; do
                        for retrieval_split in "${retrieval_splits[@]}"; do
                            echo "Running: $dataset $method $data_source $init $shots $seed $retrieval_split"

                            # Run the script and capture the output
                            output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  \
                            --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" \
                            --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --model_cfg "$model_cfg" \
                            --check_zeroshot --pre_extracted --recal_fea --skip_stage2 \
                            --folder "$output_folder")
                            
                            # Print the output to the console
                            echo "$output"

                            # Append the results to the CSV file
                            echo "$output" >> "$output_file"
                        done
                    done
                done
            done
        done
    done
done