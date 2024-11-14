#!/bin/bash

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix")
methods=("probing")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

folder="swat_imagenet_stage2_only"


# cls_inits=("random" "text" "REAL-Prompt" )
cls_inits=("REAL-Prompt")


# shot_values=(4 8 16)
shot_values=(16)

# retrieval_splits=("T2T500+T2I0.25")
# retrieval_splits=("T2T100" "T2T200" "T2T300" "T2T1000" "T2T2000")
# retrieval_splits=("T2T10")
retrieval_splits=("T2T500")


batch_size=32

# epochs=50

model_cfg="vitb32_openclip_laion400m"
# model_cfg="vitb16_openclip_laion400m"

# log_mode="file"
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

# seeds=(1 2 3)
seeds=(1)


# if dataset is imagenet, use 10 epochs
# if [ "${datasets[0]}" == "imagenet" ]; then
#     epochs=10
# fi

# epochs=(10 20 30 40 50 60 70 80 90 100)
epochs=(10)

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

# model_path="output/swat_vitb32_T2T100/output_imagenet/imagenet_cutmix_fewshot+retrieved_REAL-Prompt_16shots_seed1_10eps/stage1_model_best-epoch_10_best.pth"
model_path="output/swat_vitb32_T2T1000/output_stanford_cars/stanford_cars_cutmix_fewshot+retrieved_REAL-Prompt_16shots_seed1_50eps/stage1_model_best-epoch_50_best.pth"

# Create or clear the output file
echo "epoch,Dataset,Method,DataSource,Init,Shots,Seed,Retrieve,Stage1Acc,Stage2Acc" > "$output_file"

# Loop through all combinations and run the script
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for data_source in "${data_sources[@]}"; do
            for shots in "${shot_values[@]}"; do
                for init in "${cls_inits[@]}"; do                
                    for epoch in "${epochs[@]}"; do
                        for seed in "${seeds[@]}"; do
                            for retrieval_split in "${retrieval_splits[@]}"; do

                                echo "Running: $dataset $method $data_source $init $shots $seed $retrieval_split $epoch"

                                # model_path="output/swat_vitb32_T2T500/output_${dataset}/${dataset}_cutmix_fewshot+retrieved_REAL-Prompt_16shots_seed1_50eps/stage1_model_best-epoch_50_best.pth"

                                # Run the script and capture the output
                                output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  \
                                --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epoch" --bsz "$batch_size" \
                                --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --model_cfg "$model_cfg" \
                                --skip_stage2 \
                                --model_path "$model_path" \
                                --folder "$output_folder")
                                
                                # Print the output to the console
                                echo "$epoch,$output"

                                # Append the results to the CSV file
                                echo "$epoch,$output" >> "$output_file"
                            done
                        done
                    done
                done
            done
        done
    done
done