# Retrieve data from pretraining dataset

> Recent studies show that the LAION dataset contains CSAM content, ~~leading to its temporary removal from public access~~. See [Safety Review for LAION](https://laion.ai/notes/laion-maintenance/). We also observed that retrieved images may contain NSFW content. Please exercise caution when using this data.

We provide step-by-step instructions on how to retrieve relevant data from the OpenCLIP's pretraining dataset `LAION-400M`. In summary, we first use string matching to retrieve pretraining images whose captions contain any of the concept synonyms. We then sample from the retrievd images using prompt-to-caption (T2T) ranking to obtain 500 images for each class. The final result is a `T2T500.txt` file for each dataset, stored at `SWAT/data/{dataset}/` folder, which stores the retrieved image paths and labels.

### Easy access to our retrieved data
~~Since LAION is currently unaccessible to the public~~, for easy reproduction of our SWAT results, we provide our [matched caption files (including image URLs)](https://drive.google.com/drive/folders/1OjZ0pO4OTv3M7M85npPYhkJ3mGg6pqNO?usp=sharing) for each dataset, with which users can skip to [Step 2](#step-2-retrieve-from-laion-400m-using-string-matching) and use `laion_downloader.py` to directly download the images and continue from there. Note that some images may no longer be available on the Internet so the downloaded images maybe slightly less than the ones we had. But we expect to see the similar results as we reported in the paper.



### Example retrieved structure
The example directory structure should look like:
```
$RETRIEVED/
|–– semi-aves/
|   |–– semi-aves_retrieved_LAION400M-all_synonyms-all/
|   |–– semi-aves_vitb32_openclip_laion400m_mined.pth
|   |–– semi-aves_class_frequency-LAION400M.json
|   |–– semi-aves_downloaded_ct-LAION400M-all_synonyms-all.json
|   |–– semi-aves_metadata-all-0.0-LAION400M.map
|   |–– semi-aves_metadata-all-0.0-LAION400M.meta
|   |-- semi-aves_mined_captions-LAION400M.pkl
|   |–– semi-aves-urls-all-0.0-LAION400M.parquet
```

---
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
# here we add back the original classnames (and scientific names and common names 
# for Semi-Aves) if removed, format the output to the desired format
python format_synonyms.py

# you might want to manually check the `output/{dataset}_synonyms_filtered_final.json` 
# file and add some synonyms back if needed.

```

**As an alternative**, I used the synonyms in the metric files from [REAL](https://github.com/shubhamprshr27/NeglectedTailsVLM/tree/main/analysis/laion), and renamed for each dataset, e.g. `dtd_metrics-LAION400M.json`. Note that I have done this step for you using the commands below, and you can find the formatted metric files in the `SWAT/data/{dataset}/` folder.

```bash
# download the metric files into data/xxx folder for each dataset xxx

# format the indentation
cd SWAT/retrieval/
python format_metrics.py ../data/dtd/dtd_metrics-LAION400M.json 
```


### Step 2: retrieve from LAION-400M using string matching

Once obtained the metric files which contains the concept synonyms, we retrieve relevant pretraining images whose captions contain any of the concept synonyms. This is referred to as *string matching retrieval*. For our experiments, we retrieve from LAION-400M dataset.

- Create the `laion400.db` file from LAION's parquet files using the `create_table()` and `create_fts_table()` functions in `laion_parser.py`. Only execute this step once.
- Place the `laion400m.db` file in the `SWAT/retrieval/database` folder.
- Run string matching to obtain the matched captions.

```bash
cd SWAT/retrieval/

# string matching, ran twice for inital synonyms and manually-checked synonym list
python laion_parser.py --dataset semi-aves 
python laion_parser.py --dataset flowers102 
python laion_parser.py --dataset dtd
# adding a `texture`` prefix for retriveal gives no better performance
# python laion_parser.py --dataset dtd --prefix texture 

# add `satellite` prefix to only match captions containing "satellite" and "classname", 
# this is essential to ensure retrieving satellite images
python laion_parser.py --dataset eurosat --prefix satellite

python laion_parser.py --dataset fgvc-aircraft 
python laion_parser.py --dataset oxford_pets
python laion_parser.py --dataset food101
python laion_parser.py --dataset stanford_cars
python laion_parser.py --dataset imagenet

# or use the slurm script to run the string matching
sbatch run_stringmatching.slurm
```

- Obtain the meta data, urls, and then download the images. For small string-matched pools e.g. Semi-Aves, EuroSAT, and Aircraft, we use `--sampling all` to download all string-matched images if available. For datasets like DTD and Flowers, since their string-matched pool contains large amount of images, we set `--sampling random` to download a random subset to ease storage requirements.

```bash
# retrieve from laion400m 
python laion_downloader.py --dataset semi-aves --sampling all 
python laion_downloader.py --dataset eurosat --sampling all
python laion_downloader.py --dataset fgvc-aircraft --sampling all
python laion_downloader.py --dataset flowers102 --sampling random
python laion_downloader.py --dataset dtd --sampling random
python laion_downloader.py --dataset oxford_pets --sampling random
python laion_downloader.py --dataset food101 --sampling random
python laion_downloader.py --dataset stanford_cars --sampling random
python laion_downloader.py --dataset imagenet --sampling random

# or use the slurm script to run the retrieval
sbatch run_retrieval.slurm
```

- [Optional] delete the retrieved folder `00000/` which contains the nonessential json files.

```bash
cd $RETRIEVED/
rm -rf semi-aves/semi-aves_retrieved_LAION400M-all_synonyms-all/00000
rm -rf fgvc-aircraft/fgvc-aircraft_retrieved_LAION400M-all_synonyms-all/00000
rm -rf eurosat/eurosat_retrieved_LAION400M-all_synonyms-all/00000
rm -rf flowers102/flowers102_retrieved_LAION400M-all_synonyms-random/00000
rm -rf dtd/dtd_retrieved_LAION400M-all_synonyms-random/00000
rm -rf oxford_pets/oxford_pets_retrieved_LAION400M-all_synonyms-random/00000
rm -rf food101/food101_retrieved_LAION400M-all_synonyms-random/00000
rm -rf stanford_cars/stanford_cars_retrieved_LAION400M-all_synonyms-random/00000
rm -rf imagenet/imagenet_retrieved_LAION400M-all_synonyms-random/00000
```

### Step 3: process the downloaded data
- After obtaining the downloaded images, we format the captions for each downloaded image.

```bash
# process the meta data to the map file to get the captions for each image, 
python process_meta_map.py semi-aves
python process_meta_map.py fgvc-aircraft
python process_meta_map.py flowers102
python process_meta_map.py eurosat
python process_meta_map.py dtd

python process_meta_map.py oxford_pets
python process_meta_map.py food101
python process_meta_map.py stanford_cars
python process_meta_map.py imagenet
```
- Extract the image and text features for different sampling methods in the next step.
```bash
# extract mined images features, comment out the dataset selection in the script
python extract_mined_feature.py --dataset fgvc-aircraft --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset eurosat --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset dtd --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset flowers102 --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset semi-aves --model_cfg vitb32_openclip_laion400m

python extract_mined_feature.py --dataset oxford_pets --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset food101 --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset stanford_cars --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset imagenet --model_cfg vitb32_openclip_laion400m
```

### Step 4: sample from the downloaded data 
- We follow REAL to select top 500 images for each class using T2T ranking, i.e. ranking the cosine similarity of the prompt embedding (average of all synonyms using OpenAI templates) and the caption embedding, using the OpenCLIP ViT-B/32 model.
- The result will be a `T2T500.txt` file for each dataset, stored at `SWAT/data/{dataset}/` folder.
- Retrieval statistics are stored in `SWAT/retrieval/output/` folder.

<!-- - We use prompt-to-caption (T2T) ranking + prompt-to-image (T2I) filtering to sample 500 images for each class. 
- For classes with less than 500 downloaded images, we use all available images after T2I filtering.
- We set the T2I filtering threshold to 0.25, similar to what is used in the curation of LAION dataset.
- The result will be a `T2T500+T2I0.25.txt` file for each dataset, stored at `SWAT/data/{dataset}/` folder. -->

```bash
# run for all datasets
bash run_sampling.sh

# use T2T ranking only, this will crate a `T2T500.txt` file
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset semi-aves 
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset fgvc-aircraft 
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset eurosat 
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset dtd 
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset flowers102 
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset oxford_pets
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset food101
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset stanford_cars
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset imagenet
```

- You can play with different sampling methods, e.g. random, T2T ranking, T2I ranking, I2I ranking, T2T ranking with T2I filtering, etc.
```bash

# T2T500+T2I0.25, e.g. `T2T500+T2I0.25.txt`
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset semi-aves 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset fgvc-aircraft 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset eurosat 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset dtd 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset flowers102 
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset oxford_pets
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset food101
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset stanford_cars
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset imagenet

# random500
python sample_retrieval.py --prefix Random500 --num_samples 500 --sampling_method Random --dataset semi-aves

# t2t500
python sample_retrieval.py --prefix T2T500 --num_samples 500 --sampling_method T2T-rank --dataset semi-aves

# T2I ranking
python sample_retrieval.py --prefix T2I500 --num_samples 500 --sampling_method T2I-rank --dataset semi-aves

# I2I ranking
# need to run probing first to get the preextracted downstream images features
bash scripts/run_dataset_seed_probing.sh stanford_cars 1

python sample_retrieval.py --prefix I2I500 --num_samples 500 --sampling_method I2I-rank --dataset fgvc-aircraft

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

