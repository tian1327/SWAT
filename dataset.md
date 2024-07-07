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

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

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
- Download data from the [official repository](https://github.com/cvl-umass/semi-inat-2020) or following the `gdown` commands below
```bash
cd $DATA/semi-aves/

# train_val data
gdown https://drive.google.com/uc?id=1xsgOcEWKG9CszNNT_EXN3YB1OLPYNbf8 

# test
gdown https://drive.google.com/uc?id=1OVEA2lNJnYM5zxh3W_o_Q6K5lsNmJ9Hy

# unlabeled_ID
gdown https://drive.google.com/uc?id=1BiEkIp8yuqB5Vau_ZAFAhwp8CNoqVjRk

# unzip
tar -xzf *.gz
```
- The annotations are extracted from the [official annotation json files](https://github.com/cvl-umass/semi-inat-2020). We have reformatted and provided to you as `ltrain.txt`, `ltrain+val.txt`,`val.txt` and `test.txt` in the `SWAT/data/semi-aves/` folder.
  
The directory structure should look like:

```
semi-aves/
|–– trainval_images
|–– u_train_in
|–– test
```
