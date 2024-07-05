<div align="center">
<h1>Few-Shot Recognition via Stage-Wise Augmented Finetuning</h1>

[**Tian Liu**](https://tian1327.github.io/)<sup>1&dagger;</sup> · [**Huixin Zhang**](https://www.linkedin.com/in/huixin-zhang-a2670a229/)<sup>1</sup> · [**Shubham Parashar**](https://shubhamprshr27.github.io/)<sup>1</sup> · [**Shu Kong**](https://aimerykong.github.io/)<sup>1,2*</sup>

<sup>1</sup>Texas A&M University&emsp;&emsp;&emsp;<sup>2</sup>University of Macau
<br>
&dagger;project lead&emsp;*corresponding author

<a href="https://arxiv.org/abs/2406.11148"><img src='https://img.shields.io/badge/arXiv-SWAT-red' alt='Paper PDF'></a>
<a href='https://tian1327.github.io/SWAT/'><img src='https://img.shields.io/badge/Project_Page-SWAT-green' alt='Project Page'></a>
<!-- <a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a> -->
</div>

Our work adapt a pretrained Vision-Language Model (VLM) and retrieve relevant pretraining images to boost few-shot recognition performance.
To mitigate the problems of `domain gap` and `imbalanced distribution` of retrieved data, we propose a novel **Stage-Wise Augmented fineTuning (SWAT)** method, which outperforms previous methods by >10% in accuracy.


![teaser](assets/teaser_v7.png)

## News

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
Create conda environment and install dependencies:
```bash
git clone https://github.com/tian1327/SWAT.git 
cd SWAT

conda create -n swat python=3.8 -y
conda activate swat

conda install -y pytorch torchvision torchaudio torchmetrics -c pytorch

# need to instal the correct torchvision version
pip3 install torchvision==0.15.2

# install openclip module
pip install open_clip_torch

# install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# for retrieving images from urls
pip install img2dataset==1.2.0

# for SaliencyMix
pip3 uninstall opencv-python
pip3 install opencv-contrib-python

```

Prepare the datasets following the instructions in [dataset.md](./dataset.md).

Retrieve relevant pretraining data following the instructions in [retrieval.md](./retrieval/retrieval.md).


<!-- ### Test our model checkpoints
Download the checkpoints listed [here](#finetuned-models) and put them under the `checkpoints` directory.

```bash
# coming soon

``` -->

### Running SWAT

You can run SWAT by using the bash script `run_dataset_seed.sh` (recommended) or the python `main.py` script.
For example, using the bash script:
```bash
# 1. update the options in run_dataset_seed.sh, this can be used to run a batch of experiments
# 2. run the bash script in command line
bash run_dataset_seed.sh semi-aves 1
```

For example using the python `main.py` script with more fine-grained controls:
```bash
# run finetune on few-shot on semi-aves dataset with 4-shot, seed 1
python main.py --dataset semi-aves --method finetune --data_source fewshot --cls_init REAL-Prompt --shots 4 --seed 1 --epochs 50 --bsz 32 --log_mode both --retrieval_split T2T500+T2I0.25.txt --model_cfg vitb32_openclip_laion400m --folder output/test_finetune_on_fewshot

# run SWAT on semi-aves dataset with 4-shot, seed 1
python main.py --dataset semi-aves --method cutmix --data_source mixed --cls_init REAL-Prompt --shots 4 --seed 1 --epochs 50 --bsz 32 --log_mode both --retrieval_split T2T500+T2I0.25.txt --model_cfg vitb32_openclip_laion400m --folder output/test_swat

```


## Acknowledgment
This code base is developed with some references on the following projects. We sincerely thanks the authors for open-sourcing their projects.

- REAL: https://github.com/shubhamprshr27/NeglectedTailsVLM
- Cross-modal few-shot adaptation: https://github.com/linzhiqiu/cross_modal_adaptation
- OpenCLIP: https://github.com/mlfoundations/open_clip 

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{liu2024few,
  title={Few-Shot Recognition via Stage-Wise Augmented Finetuning},
  author={Liu, Tian and Zhang, Huixin and Parashar, Shubham and Kong, Shu},
  journal={arXiv preprint arXiv:2406.11148},
  year={2024}
}

@inproceedings{parashar2024neglected,
  title={The Neglected Tails in Vision-Language Models},
  author={Parashar, Shubham and Lin, Zhiqiu and Liu, Tian and Dong, Xiangjue and Li, Yanan and Ramanan, Deva and Caverlee, James and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

```