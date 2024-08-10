# Retrieve data from pretraining dataset

### Step 1: query GPT-3.5/4 for concept synonyms
From my experiences, GPT-4 usually gives more diverse synonyms than GPT-3.5, 
including names that are in other languages e.g. Chinese, Japaneses, etc. 
and some unformatted text. To avoid this issue and get English synonyms only, 
I appended the following instruction in the prompt:
`Don't give any other text. Give me English names only.` 
See `query_synonyms.py` for more details.

```bash 
# I experimented following code with Semi-Aves using GPT4
cd SWAT/retrieval/query_synonyms/

# paste your openai key in the openai_key.txt file

# set the target dataset and GPT model to use in this script, 
# query aves with most common name, modified the prompt
# select the target datasets in `targets` list
python query_synonyms.py

# run text classification with OpenCLIP
python clip_text_filtering.py

# Since OpenCLIP may not understand some classnames or scientific names very well, 
# here we add back the original classnames (and scientific names and common names for Semi-Aves) if removed, 
# format the output to the desired format
python format_synonyms.py

# you might want to manually check the `output/{dataset}_synonyms_filtered_final.json` file and add some synonyms back if needed.

```

As an alternative, I used the synonyms in the metric files of REAL [repo](https://github.com/shubhamprshr27/NeglectedTailsVLM/tree/main/analysis/laion).

```bash
# download the metric files into data/xxx folder for each dataset xxx
cd data/

# format the indentation
python format_metrics.py dtd/dtd_metrics-LAION400M.json 
```



### Step 2: retrieve from laion400m

[to be updated !!!!]

```bash
cd retrieval/

# string matching, ran twice for inital synonyms and manually-checked synonym list
python laion_parser.py --dataset semi-aves # need to replace the keywords `alternates` to `synonyms_final` ---> no need 
python laion_parser.py --dataset flowers102 
python laion_parser.py --dataset dtd
# python laion_parser.py --dataset dtd --prefix texture # no better
python laion_parser.py --dataset eurosat --prefix satellite # add prefix to only match captions containing "satellite" and "classname"
python laion_parser.py --dataset fgvc-aircraft 

# retrieve from laion400m
python laion_downloader.py --dataset semi-aves --sampling all 
python laion_downloader.py --dataset flowers102 --sampling random
python laion_downloader.py --dataset dtd --sampling random
# python laion_downloader.py --dataset dtd --sampling all
python laion_downloader.py --dataset eurosat --sampling all
python laion_downloader.py --dataset fgvc-aircraft --sampling all

# or use the slurm script to run the retrieval
sbatch run_retrieval.slurm

# need to manually delete the retrieved folder 00000 which contains the json files
rm -rf semi-aves/semi-aves_retrieved_LAION400M-all_synonyms-all/00000
rm -rf fgvc-aircraft/fgvc-aircraft_retrieved_LAION400M-all_synonyms-all/00000
rm -rf eurosat/eurosat_retrieved_LAION400M-all_synonyms-all/00000
rm -rf flowers102/flowers102_retrieved_LAION400M-all_synonyms-random/00000
rm -rf dtd/dtd_retrieved_LAION400M-all_synonyms-all/00000
```

### Step 3: post-process the downloaded data

```bash
# process the meta data to the map file to get the captions for each image, 
# be careful on the fn and fn_out, do not use the same name! 
# here I accidentally used the same name and delete the old meta file by mistake for semi-aves dataset
python process_meta_map.py 

# extract mined images features, comment out the dataset selection in the script
python extract_mined_feature.py --dataset fgvc-aircraft 
python extract_mined_feature.py --dataset eurosat
python extract_mined_feature.py --dataset dtd
python extract_mined_feature.py --dataset flowers102
python extract_mined_feature.py --dataset semi-aves

```

### Step 4: run the T2T sampling with T2I filtering
The following command will generate the label file for the retrieved data, e.g. `T2T500+T2I0.25.txt`
```bash
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method t2t-rank-t2i-tshd --dataset semi-aves 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method t2t-rank-t2i-tshd --dataset fgvc-aircraft 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method t2t-rank-t2i-tshd --dataset eurosat 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method t2t-rank-t2i-tshd --dataset dtd 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method t2t-rank-t2i-tshd --dataset flowers102 
```

Run T2Tranking
```bash
# t2t500
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method t2t-rank --dataset semi-aves

# random500
python sample_retrieval.py --prefix Random500 --num_samples 500 --sampling_method random --dataset semi-aves

# T2I ranking
python sample_retrieval.py --prefix T2I500 --num_samples 500 --sampling_method t2i-rank --dataset semi-aves

# I2I ranking
# need to run probing first to get the preextracted features
python main.py --dataset eurosat --method probing --data_source fewshot --cls_init REAL-Prompt --shots 16 --seed 1 --epochs 10 --pre_extracted True --recal_fea  --cls_init REAL-Prompt --skip_stage3 --folder output_probing

python sample_retrieval.py --prefix I2I500 --num_samples 500 --sampling_method i2i-rank --dataset fgvc-aircraft


# I2T-ranking
python sample_retrieval.py --prefix I2T-rank500 --num_samples 500 --sampling_method I2T-rank --dataset semi-aves

# T2T ranking + I2T filtering
python sample_retrieval.py --prefix T2T500+I2T0.25 --num_samples 500 --sampling_method T2T-rank-I2T-tshd --dataset semi-aves

# T2T ranking + I2I filtering
python sample_retrieval.py --prefix T2T500+I2I0.5 --num_samples 500 --sampling_method T2T-rank-I2I-tshd --dataset semi-aves
# update the threshold in add_t2t_ranked_t2i_tshd_to_split() in sample_retrieval.py
python sample_retrieval.py --prefix T2T500+I2I0.65 --num_samples 500 --sampling_method T2T-rank-I2I-tshd --dataset eurosat

# I2T-filtering
# python sample_retrieval.py --prefix I2T-tshd500 --num_samples 500 --sampling_method I2T-tshd --dataset semi-aves

```

### Step 5: prepare downstream datasets

1. Mostly follow the instruction [here](https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/DATASETS.md)

```bash
# aircraft
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz

# eurosat
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1

# dtd
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

# flowers102
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

```

2. Prepare the fewshotX-seedX.text, and test.txt files for each dataset.
```bash
python prepare_datasets_labels.py
```


### Step 7: run standard finetuning
1. Prepare the few-shot annotation files for each dataset
```bash
python prepare_fewshot_txt.py
```

2. run standard finetuning on mixed data

