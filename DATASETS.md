# How to install downstream datasets

*The dataset installation instruction is modified from official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md), with some references of updates from [CMLP](https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/DATASETS.md).*


We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
$DATA/
|–– semi-aves/
|–– flowers102/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/
|–– oxford_pets/
|–– stanford_cars/
|–– food101/
|–– imagenet/

$RETRIEVED/
|–– semi-aves/
|–– flowers102/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/
|–– oxford_pets/
|–– stanford_cars/
|–– food101/
|–– imagenet/
```

Update the `config.yml` with the path to the datasets and retrieved data.
```yaml
dataset_path: /scratch/group/real-fs/dataset/
retrieved_path: /scratch/group/real-fs/retrieved/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Below we provide instructions to prepare the downstream datasets used in our experiments. Please refer to [RETRIEVAL.md](retrieval/RETRIEVAL.md) for instructions on how to setting up the retrieved datasets.

Datasets list:

- [Semi-Aves](#semi-aves)

- [Flowers102](#flowers102)
- [FGVC-Aircraft](#fgvcaircraft)
- [EuroSAT](#eurosat)
- [DTD](#dtd)

- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Food101](#food101)

- [ImageNet](#imagenet)


<!-- - [Caltech101](#caltech101)
- [SUN397](#sun397)
- [UCF101](#ucf101) -->



The instructions to prepare each dataset are detailed below. 
To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us. For few-shot training data, we sample from given training data with 3 random seeds. The splits are available in the `SWAT/data/{dataset}/` folder.


### Semi-Aves

- Create a folder named `semi-aves/` under `$DATA`.
- Download data from the [official repository](https://github.com/cvl-umass/semi-inat-2020) or following the `wget` commands below
```bash
cd $DATA/semi-aves/

# train_val data
wget https://drive.google.com/uc?id=1xsgOcEWKG9CszNNT_EXN3YB1OLPYNbf8 

# test
wget https://drive.google.com/uc?id=1OVEA2lNJnYM5zxh3W_o_Q6K5lsNmJ9Hy

# unlabeled_ID
wget https://drive.google.com/uc?id=1BiEkIp8yuqB5Vau_ZAFAhwp8CNoqVjRk

# unzip
tar -xzf *.gz
```
- The annotations are extracted from the [official annotation json files](https://github.com/cvl-umass/semi-inat-2020). We have reformatted labels and provided to you as `ltrain.txt`, `ltrain+val.txt`,`val.txt` and `test.txt` in the `SWAT/data/semi-aves/` folder.
  
The directory structure should look like:

```
semi-aves/
|–– trainval_images
|–– u_train_in
|–– test
```

### Flowers102

- Create a folder named `flowers102/` under `$DATA`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
```
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/flowers102/` folder.

The directory structure should look like:

```
flowers102/
|–– jpg
|–– imagelabels.mat
|–– cat_to_name.json
```

### FGVC-Aircraft
- Create a folder named `fgvc-aircraft/` under `$DATA`.
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
```
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `fgvc-aircraft-2013b/data/` to `$DATA/fgvc-aircraft`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/fgvc-aircraft/` folder.

The directory structure should look like:
```
fgvc_aircraft/
|–– fgvc-aircraft-2013b/
    |–– data/
```

### EuroSAT
- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
```bash
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
```
- Renmae the extracted folder `2750` to `EuroSAT_RGB`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/eurosat/` folder.

The directory structure should look like:
```
eurosat/
|–– EuroSAT_RGB/
```

### DTD
- Create a folder named `dtd/` under `$DATA`.
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
```
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/dtd/` folder.

The directory structure should look like:
```
dtd/
|–– dtd/
    |–– images/
    |–– imdb/
    |–– labels/
```


### OxfordPets
- Create a folder named `oxford_pets/` under `$DATA`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/oxfordpets/` folder.

The directory structure should look like:
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `$DATA`.
- In case the following link breaks, download dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).
- Download `car_devkit.tgz`
```bash
wget https://github.com/pytorch/vision/files/11644847/car_devkit.tgz
tar -xzvf car_devkit.tgz
```
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

- ~~Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.~~
- ~~Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.~~
- ~~Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.~~
- ~~Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.~~


The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_annos.mat
|–– cars_train\
|–– split_zhou_StanfordCars.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### Caltech101
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from https://data.caltech.edu/records/mzrjq-6wc02 and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– Annotations/
|–– split_zhou_Caltech101.json
```

### SUN397
- Create a folder named `sun397/` under `$DATA`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `$DATA/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### UCF101
- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Download `split_ImageNet.json` [(google drive link)](https://drive.google.com/file/d/1SvPIN6iV6NP2Oulj19a869rBXrB5SNFo/view) to this folder. (Note that the original CoOp does not make a train/val split.)
- Create `images/` under `imagenet/`.
-Option 1: download dataset from [Huggingface repo](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

```bash
# create a HF account and get the API token, then run the following command
wget --header="Authorization: Bearer YOUR_HUGGINGFACE_TOKEN" https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/train_images_0.tar.gz

mkdir train
tar -xzf train_images_0.tar.gz -C train/

mkdir val
tar -xzf val_images.tar.gz -C val/

```

- Option 2: Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. 
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

The directory structure should look like
```
imagenet/
|–– classnames.txt
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).




---

## Prepare the dataset labels
- We have already prepared the `train.txt`, `val.txt` and `test.txt` files for each dataset, using the same splits as in [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
- In case you are interested, we use the following script to prepare the dataset labels.
```bash
cd SWAT/
python prepare_datasets_labels.py
```

## Prepare the few-shot labels
- Prepare the few-shot annotation files for each dataset, using 3 random seeds.
- We have already prepared the `fewshot{4/8/16}_seed{1/2/3}.txt` files for each dataset in `SWAT/data/{dataset}/` folder.
```bash
python prepare_fewshot_txt.py
```


