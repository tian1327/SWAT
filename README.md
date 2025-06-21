<div align="center">
<h1>Few-Shot Recognition via Stage-Wise<br>Retrieval-Augmented Finetuning</h1>

[**Tian Liu**](https://tian1327.github.io/)<sup>1</sup> · [**Huixin Zhang**](https://www.linkedin.com/in/huixin-zhang-a2670a229/)<sup>1</sup> · [**Shubham Parashar**](https://shubhamprshr27.github.io/)<sup>1</sup> · [**Shu Kong**](https://aimerykong.github.io/)<sup>2</sup>

<sup>1</sup>Texas A&M University&emsp;&emsp;&emsp;<sup>2</sup>University of Macau
<br>
<!-- &dagger;project lead&emsp;*corresponding author -->

<a href="https://arxiv.org/abs/2406.11148"><img src='https://img.shields.io/badge/arXiv-SWAT-red' alt='Paper PDF'></a>
<a href='https://tian1327.github.io/SWAT/'><img src='https://img.shields.io/badge/Project_Page-SWAT-green' alt='Project Page'></a>
<!-- <a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a> -->
</div>

Our work adapts a pretrained Vision-Language Model (VLM) and retrieves relevant pretraining images to solve few-shot recognition problem.
To mitigate the `domain gap` and `imbalanced distribution` problems of retrieved data, we propose a novel **Stage-Wise retrieval-Augmented fineTuning (SWAT)** method, which outperforms previous few-shot recognition methods by >6% in accuracy across nine benchmark datasets.


![teaser](assets/swat.png)

## News
<!-- - **2024-11-26:** updated [arXiv paper](), including more datasets. -->
- **2025-05-27:** SWAT is accepted to 4th CVinW and FGVC12 workshops at CVPR'25! 
- **2025-02-26:** SWAT is accepted to CVPR 2025! ;)
- **2025-01-18:** We provide access to our retrieved data through URLs. See [RETRIEVAL.md](./retrieval/RETRIEVAL.md).
- **2024-11-24:** Updated code base to include more datasets.
- **2024-08-22:** Retrieval code released, see [RETRIEVAL.md](./retrieval/RETRIEVAL.md).
- **2024-07-05:** SWAT finetuning code released.
- **2024-06-28:** [project page](https://tian1327.github.io/SWAT/) launched.
- **2024-06-17:** [arXiv paper](https://arxiv.org/abs/2406.11148) released.


<!-- ## Finetuned Models

We provide SWAT finetuned model (based on OpenCLIP ViT-B/32) for each dataset experimented in the paper:

| Dataset | Size | Checkpoint |
|:-|:-|:-:|
| Semi-Aves |  | [Download]() |
| Flowers102 |  | [Download]() |
| FGVC-Aircraft |  | [Download]() |
| EuroSAT |  | [Download]() |
| DTD |  | [Download]() | -->


## Usage

### Prepraration
Create conda environment and install dependencies following the instructions in [ENV.md](./ENV.md).

Prepare the datasets following the instructions in [DATASETS.md](./DATASETS.md).

Retrieve relevant pretraining data following the instructions in [RETRIEVAL.md](./retrieval/RETRIEVAL.md).


<!-- ### Test our model checkpoints
Download the checkpoints listed [here](#finetuned-models) and put them under the `checkpoints` directory.

```bash
# coming soon

``` -->

### Running SWAT

You can run SWAT and finetune on few-shot using the following bash scripts.

<!-- You can run SWAT by using either the bash scripts `run_dataset_seed_xxx.sh` (recommended) or the python `main.py` script.
For example, using the bash scripts: -->
```bash
# 1. check the options in run_dataset_seed_xxx.sh,
#    this can be used to run a batch of experiments.
# 2. run the corresponding bash script in command line
# Usage: bash scripts/run_dataset_seed_xxx.sh <dataset> [seed]

# finetune on few-shot, seed 1
bash scripts/run_dataset_seed_finetune_fewshot.sh semi-aves 1

# finetune on few-shot with CutMix, 3 seeds
bash scripts/run_dataset_seed_finetune_fewshot_cutmix.sh semi-aves

# swat
bash scripts/run_dataset_seed_SWAT.sh semi-aves 1
```

<!-- For example, using the python `main.py` script with more explicit fine-grained controls:
```bash
# run finetune on few-shot on semi-aves dataset with 4-shot, seed 1
python main.py --dataset semi-aves --method finetune --data_source fewshot --cls_init REAL-Prompt --shots 4 --seed 1 --epochs 50 --bsz 32 --log_mode both --retrieval_split T2T500+T2I0.25.txt --model_cfg vitb32_openclip_laion400m --folder output/finetune_on_fewshot

# run SWAT on semi-aves dataset with 4-shot, seed 1
# note that SWAT uses `--method cutmix` and `--data_source fewshot+retrieved`
python main.py --dataset semi-aves --method cutmix --data_source fewshot+retrieved --cls_init REAL-Prompt --shots 4 --seed 1 --epochs 50 --bsz 32 --log_mode both --retrieval_split T2T500+T2I0.25.txt --model_cfg vitb32_openclip_laion400m --folder output/swat -->

The results of the experiments will be saved in the `result` directory. The detailed logs, models, and scores etc. will be saved in the `output` directory.

### Running other baselines
Below we provide the commands to run the zero-shot and few-shot baselines in the paper. Update the `model_cfg` option in the bash scripts to use different models.

Zero-shot methods:
```bash
# OpenCLIP zero-shot
bash scripts/run_dataset_zeroshot.sh semi-aves

# REAL-Prompt
bash scripts/run_dataset_REAL-Prompt.sh semi-aves

# REAL-Linear
# take the WSFT accuracy with alpha=0.5
# find the line: `Alpha:0.5, Val Acc: 48.671, Test Acc: 48.562`
bash scripts/run_dataset_REAL-Linear.sh semi-aves

```

Few-shot methods:
```bash
# Cross-modal Linear Probing (CMLP)
bash scripts/run_dataset_seed_CMLP.sh semi-aves 1
```

For [CLAP](https://github.com/jusiro/CLAP), we use the provided code but replace the model from CLIP to OpenCLIP. Our implementation can be found in [CLAP-tian](https://github.com/tian1327/CLAP-tian) with [instructions](https://github.com/tian1327/CLAP-tian/blob/main/tian_log.md).


## Acknowledgment
This code base is developed with some references on the following projects. We sincerely thank the authors for open-sourcing their projects.

- [REAL](https://github.com/shubhamprshr27/NeglectedTailsVLM)
- [Cross-modal few-shot adaptation](https://github.com/linzhiqiu/cross_modal_adaptation)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

## Citation

If you find our project useful, please consider citing:

```bibtex
@inproceedings{liu2025few,
  title={Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning},
  author={Liu, Tian and Zhang, Huixin and Parashar, Shubham and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{parashar2024neglected,
  title={The Neglected Tails in Vision-Language Models},
  author={Parashar, Shubham and Lin, Zhiqiu and Liu, Tian and Dong, Xiangjue and Li, Yanan and Ramanan, Deva and Caverlee, James and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

```
