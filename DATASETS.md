# How to install datasets

*The dataset download instruction is modified from official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).*

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
$DATA/
|–– semi-aves/
|–– flowers102/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/

$RETRIEVED/
|–– semi-aves/
|–– flowers102/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/
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

The instructions to prepare each dataset are detailed below. 
<!-- To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us. -->

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
- Renmae the extracted folder `2750` to `EuroSAT_RGB`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/eurosat/` folder.

The directory structure should look like:
```
eurosat/
|–– EuroSAT_RGB/
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWAT/data/dtd/` folder.

The directory structure should look like:
```
dtd/
|–– dtd/
    |–– images/
    |–– imdb/
    |–– labels/
```

