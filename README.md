# RTU-Net

## Requirements
- [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus#compute)
- Linux X86 (tested on Ubuntu 18.04.6 LTS)
- [anaconda](https://www.anaconda.com/download) (tested with anaconda 23.1.0)
- Pytorch 2.1.0, torchvision 0.16.0 (the code is tested with python=3 .8, cuda=12.0)

## Installation
##### Clone repository
Clone and navigate to this repository
```
git clone [https://github.com/linbingzhi/RTU-Net](https://github.com/linbingzhi/RTU-Net)
cd RTU-Net
```

##### Install RTU-Net conda environment

```
conda env create -f environment.yml
conda activate RTU-Net
```
You could also pip the environment by running the following code.
```
pip install -r requirements.txt
```

##### Install time
The install time greatly depends on download speed (several hundred MB).<br>
🕐 Without download (or very fast download), the [installation](#install-hylfm-conda-environment) takes around **3 min**.

## Demo
##### Activate RTU-Net conda environment
```
conda activate RTU-Net
```

##### [optional] Choose a CUDA device
A cuda device may be selected before running RTU-Net (default 0), e.g.
```
export CUDA_VISIBLE_DEVICES=3
```

#### Train RTU-Net on beads
```
python train.py
```
🕐 Excluding download time, this training configuration runs for approximately **6 hours** (on a A100 PCIe 40GB). Note that the network will likely not have fully converged; increase `max_epochs` to allow for longer training.


#### Test RTU-Net on beads (no previous training required)
To download and test RTU-Net dataset run
```
python predict.py
```
🕐 Excluding download time, this test configuration runs for approximately **5 min** in total with 0.1 s per sample (on a A100 PCIe 40GB). Most time is spend on computing metrics.

