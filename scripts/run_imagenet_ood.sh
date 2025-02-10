#!/bin/bash

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix")
methods=("finetune")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

folder="ImageNet_OOD_zeroshot_vitb16"
# folder="ImageNet_OOD_REAL-Prompt_vitb16"
# folder="ImageNet_OOD_REAL-Linear_vitb16"



# cls_inits=("random" "text" "REAL-Prompt" )
# cls_inits=("REAL-Prompt")
cls_inits=("text")



# shot_values=(4 8 16)
shot_values=(16)

retrieval_splits=("T2T500")

batch_size=32

# epochs=50

# model_cfg="vitb32_openclip_laion400m"
model_cfg="vitb16_openclip_laion400m"

log_mode="both"

#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

# Check if command-line arguments were provided
# if [ "$#" -ge 1 ]; then
#     datasets=("$1")  # Use the provided command-line argument for the dataset
# else
#     echo "Usage: $0 <dataset> [seed]"
# fi

datasets=("imagenet")
# seeds=(1 2 3)
seeds=(1)


# if dataset is imagenet, use 10 epochs
# if [ "${datasets[0]}" == "imagenet" ]; then
#     epochs=10
# fi

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

                                # ImageNet SWAT checkpoint
                                # model_path="output/swat_vitb16/output_imagenet/imagenet_cutmix_fewshot+retrieved_REAL-Prompt_16shots_seed1_10eps/stage2_model_best-epoch_10_best.pth"

                                # REAL-Linear
                                # model_path = "output/REAL-Linear_vitb16/output_imagenet/imagenet_REAL-Linear_retrieved_REAL-Prompt_16shots_seed1_10eps/stage1_model_best-epoch_10_best.pth"

                                # FTFS w/ CutMix
                                # model_path = "output/FTFS-cutmix_vitb16/output_imagenet/imagenet_cutmix_fewshot_REAL-Prompt_16shots_seed1_10eps/stage1_model_best-epoch_10_best.pth"

                                # FT on retrieved. Note this is ViT-B/32
                                # model_path = "output/FT_retrieved_vitb32/output_imagenet/imagenet_finetune_retrieved_REAL-Prompt_16shots_seed1_10eps/stage1_model_best-epoch_10_best.pth"

                                # FT on retrieved + stage 2 classifier retraining. Note this is ViT-B/32
                                # model_path = "output/FT_retrieved_vitb32/output_imagenet/imagenet_finetune_retrieved_REAL-Prompt_16shots_seed1_10eps/stage2_model_best-epoch_10_best.pth"


                                echo "Running: $dataset $method $data_source $init $shots $seed $retrieval_split $epoch"

                                # Run the script and capture the output
                                output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  \
                                --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epoch" --bsz "$batch_size" \
                                --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --model_cfg "$model_cfg" \
                                --test_imagenet_ood --skip_stage2 \
                                --folder "$output_folder" \
                                --check_zeroshot
                                # --model_path "$model_path" \
                                )

                            done
                        done
                    done
                done
            done
        done
    done
done