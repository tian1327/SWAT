You can set up the environment by following the instructions below:

Easy way:
```bash
git clone https://github.com/tian1327/SWAT.git 
cd SWAT

conda create --name swat --file requirements.txt

conda activate swat
```

Or funnier way ;)
```bash
conda create -n swat python=3.8 -y
conda activate swat

conda install -y pytorch torchvision torchaudio torchmetrics -c pytorch

# need to install the correct torchvision version
pip3 install torchvision==0.15.2

# install openclip module
pip install open_clip_torch

# install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# for retrieving images from urls
pip install img2dataset

# for SaliencyMix
pip3 uninstall opencv-python
pip3 install opencv-contrib-python

```