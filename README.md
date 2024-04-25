# WaveMix
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-byclass)](https://paperswithcode.com/sota/image-classification-on-emnist-byclass?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-bymerge)](https://paperswithcode.com/sota/image-classification-on-emnist-bymerge?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-letters)](https://paperswithcode.com/sota/image-classification-on-emnist-letters?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-inat2021-mini)](https://paperswithcode.com/sota/image-classification-on-inat2021-mini?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/scene-classification-on-places365-standard)](https://paperswithcode.com/sota/scene-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-galaxy10-decals)](https://paperswithcode.com/sota/image-classification-on-galaxy10-decals?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-caltech-256)](https://paperswithcode.com/sota/image-classification-on-caltech-256?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-places365-standard)](https://paperswithcode.com/sota/image-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-svhn)](https://paperswithcode.com/sota/image-classification-on-svhn?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=wavemix-lite-a-resource-efficient-neural)

 
  


## Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform 

### WaveMix Architecture
![image](https://user-images.githubusercontent.com/15833382/226090639-b4571494-7d2d-4bcb-81e3-127916339dfe.png)

### WaveMix-Lite
![image](https://user-images.githubusercontent.com/15833382/226090664-d844e4f1-854a-43b3-8106-78307f187fe8.png)

We propose WaveMix– a novel neural architecture for computer vision that is resource-efficient yet generalizable and scalable. WaveMix networks achieve comparable or better accuracy than the state-of-the-art convolutional neural networks, vision transformers, and token mixers for several tasks, establishing new benchmarks for segmentation on Cityscapes; and for classification on Places-365, f ive EMNIST datasets, and iNAT-mini. Remarkably, WaveMix architectures require fewer parameters to achieve these benchmarks compared to the previous state-of-the-art. Moreover, when controlled for the number of parameters, WaveMix requires lesser GPU RAM, which translates to savings in time, cost, and energy. To achieve these gains we used multi-level two-dimensional discrete wavelet transform (2D-DWT) in WaveMix blocks, which has the following advantages: (1) It reorganizes spatial information based on three strong image priors– scale-invariance, shift-invariance, and sparseness of edges, (2) in a lossless manner without adding parameters, (3) while also reducing the spatial sizes of feature maps, which reduces the memory and time required for forward and backward passes, and (4) expanding the receptive field faster than convolutions do. The whole architecture is a stack of self-similar and resolution-preserving WaveMix blocks, which allows architectural f lexibility for various tasks and levels of resource availability.


| Task                  | Dataset     | Metric   | Value  |
|-----------------------|-------------|----------|--------|
| Semantic Segmentation | Cityscapes  | Single-scale mIoU     | 83.040% (SOTA) |
| Image Classification  | ImageNet-1k | Accuracy | 75.32% |
| Image Classification  | CIFAR-10    | Accuracy | 95.98% |
| Image Classification  | Galaxy 10 DECals | Accuracy | 95.42% (SOTA) | 


### Parameter Efficiency
| Task                         | Model                                           | Parameters |
|------------------------------|-------------------------------------------------|------------|
| 99% Accu. in MNIST           | WaveMix Lite-8/10                               | 3566       |
| 90% Accu. in Fashion MNIST   | WaveMix Lite-8/5                                | 7156       |
| 80% Accu. in CIFAR-10        | WaveMix Lite-32/7                               | 37058      |
| 90% Accu. in CIFAR-10        | WaveMix Lite-64/6                               | 520106     |   

The high parameter efficiency is obtained by replacing Deconvolution layers with Upsampling

This is an implementation of code from the following papers : [Openreview Paper](https://openreview.net/forum?id=tBoSm4hUWV), [ArXiv Paper 1](https://arxiv.org/abs/2203.03689), [ArXiv Paper 2](https://arxiv.org/abs/2205.14375)

## Install

```bash
$ pip install wavemix
```

## Usage
### Semantic Segmentation

```python
import torch, wavemix
from wavemix.SemSegment import WaveMix
import torch

model = WaveMix(
    num_classes= 20, 
    depth= 16,
    mult= 2,
    ff_channel= 256,
    final_dim= 256,
    dropout= 0.5,
    level=4,
    stride=2
)

img = torch.randn(1, 3, 256, 256)

preds = model(img) # (1, 20, 256, 256)
```

### Image Classification

```python
import torch, wavemix
from wavemix.classification import WaveMix
import torch

model = WaveMix(
    num_classes= 1000, 
    depth= 16,
    mult= 2,
    ff_channel= 192,
    final_dim= 192,
    dropout= 0.5,
    level=3,
    patch_size=4,
)
img = torch.randn(1, 3, 256, 256)

preds = model(img) # (1, 1000)
```

### Single Image Super-resolution

```python
import wavemix, torch
from wavemix.sisr import WaveMix

model = WaveMix(
    depth = 4,
    mult = 2,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.5,
    level=1,
)

img = torch.randn(1, 3, 256, 256)
out = model(img) # (1, 3, 512, 512)
```

### To use a single Waveblock

```python
import wavemix, torch
from wavemix import Level1Waveblock

```

## Parameters

- `num_classes`: int.  
Number of classes to classify/segment.
- `depth`: int.  
Number of WaveMix blocks.
- `mult`: int.  
Expansion of channels in the MLP (FeedForward) layer. 
- `ff_channel`: int.  
No. of output channels from the MLP (FeedForward) layer. 
- `final_dim`: int.  
Final dimension of output tensor after initial Conv layers. Channel dimension when tensor is fed to WaveBlocks.
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout rate. 
- `level`: int.  
Number of levels of 2D wavelet transform to be used in Waveblocks. Currently supports levels from 1 to 4.
- `stride`: int.  
Stride used in the initial convolutional layers to reduce the input resolution before being fed to Waveblocks. 
- `initial_conv`: str.  
Deciding between strided convolution or patchifying convolutions in the intial conv layer. Used for classification. 'pachify' or 'strided'.
- `patch_size`: int.  
Size of each non-overlaping patch in case of patchifying convolution. Should be a multiple of 4.


#### Cite the following papers 
```
@misc{
p2022wavemix,
title={WaveMix: Multi-Resolution Token Mixing for Images},
author={Pranav Jeevan P and Amit Sethi},
year={2022},
url={https://openreview.net/forum?id=tBoSm4hUWV}
}

@misc{jeevan2022wavemix,
    title={WaveMix: Resource-efficient Token Mixing for Images},
    author={Pranav Jeevan and Amit Sethi},
    year={2022},
    eprint={2203.03689},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{jeevan2023wavemix,
      title={WaveMix: A Resource-efficient Neural Network for Image Analysis}, 
      author={Pranav Jeevan and Kavitha Viswanathan and Anandu A S and Amit Sethi},
      year={2023},
      eprint={2205.14375},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
