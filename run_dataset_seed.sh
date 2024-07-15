#!/bin/bash

# Define arrays of values for each parameter

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix")
# methods=("finetune") # this is strandard finetune
# methods=("cutmix") # this SWAT
methods=("cutmix-fs") # this SWAT



# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
# data_sources=("fewshot")
data_sources=("fewshot+retrieved")
# data_sources=("fewshot+unlabeled")
# data_sources=("fewshot+retrieved+unlabeled")
# data_sources=("ltrain+val+unlabeled")
# data_sources=("ltrain+val+unlabeled+retrieved")


folder="cutmix-fs_strict_batch_mix"
# folder="finetune_on_fewshot"
# folder="finetune_on_fewshot_CutMix"
# folder="ft_fewshot+unlabeled_in"
# folder="swat_fewshot+retr+unlabeled_in"
# folder="ft_ltrain+val+unlabeled_oracle+retrieved_noCutMix"


# cls_inits=("random" "text" "REAL-Prompt" )
cls_inits=("REAL-Prompt")


# shot_values=(4 8 16)
shot_values=(16)
# shot_values=(4)


# retrieval_splits=("T2T100+T2I0.25" "T2T300+T2I0.25" "T2T1000+T2I0.25" "T2T2000+T2I0.25")
# retrieval_splits=("Random500" "T2T500" "T2I500" "I2I500")
# retrieval_splits=("I2T-rank500" "T2T500+I2T0.25" "T2T500+I2I0.5")
# retrieval_splits=("T2T500+I2I0.65")
retrieval_splits=("T2T500+T2I0.25")


# unlabeled_splits=("u_train_in_oracle.txt" "u_train_in.txt")
unlabeled_splits=("u_train_in_oracle.txt")

batch_size=32
# batch_size=256


epochs=50
# epochs=1 # for quick testing only


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

if [ "$#" -ge 2 ]; then
    seeds=("$2")  # Use the provided command-line argument for the seed
else
    seeds=(1 2 3)
fi


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
echo "Dataset,Method,DataSource,Init,Shots,Seed,Retrieve,Stage1Acc,Stage2Acc" > "$output_file"

# Loop through all combinations and run the script
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for data_source in "${data_sources[@]}"; do
            for shots in "${shot_values[@]}"; do
                for init in "${cls_inits[@]}"; do                
                    for seed in "${seeds[@]}"; do
                        for retrieval_split in "${retrieval_splits[@]}"; do
                            for unlabeled_split in "${unlabeled_splits[@]}"; do
                                echo "Running: $dataset $method $data_source $init $shots $seed $retrieval_split $unlabeled_in_split"

                                # Run the script and capture the output
                                output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --unlabeled_split "$unlabeled_split" --model_cfg "$model_cfg" --folder "$output_folder")
                                
                                # Print the output to the console
                                echo "$output"

                                # Append the results to the CSV file
                                echo "$output" >> "$output_file"
                            done

                            # ablate retrieval size
                            # output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epochs" --log_mode file --retrieval_split "${retrieval_split}.txt" --prefix "$retrieval_split" --folder output_ablate_retrieval_size)
                            
                            # ablate the methods/MSDA
                            # output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epochs" --log_mode file --retrieval_split "${retrieval_split}.txt" --folder output_ablate_MSDA)

                            # run CMLP for different shots and seeds
                            # output=$(python main.py --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" --shots "$shots" --seed "$seed" --epochs "$epochs" --log_mode file --pre_extracted True --recal_fea --model_cfg vitb16_openclip_laion400m --folder output_CMLP_vitb16_50eps)                        
                        done
                    done
                done
            done
        done
    done
done