# WaveMix
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-byclass)](https://paperswithcode.com/sota/image-classification-on-emnist-byclass?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-bymerge)](https://paperswithcode.com/sota/image-classification-on-emnist-bymerge?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-letters)](https://paperswithcode.com/sota/image-classification-on-emnist-letters?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-inat2021-mini)](https://paperswithcode.com/sota/image-classification-on-inat2021-mini?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/scene-classification-on-places365-standard)](https://paperswithcode.com/sota/scene-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-caltech-256)](https://paperswithcode.com/sota/image-classification-on-caltech-256?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-places365-standard)](https://paperswithcode.com/sota/image-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-svhn)](https://paperswithcode.com/sota/image-classification-on-svhn?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=wavemix-lite-a-resource-efficient-neural)
 
  


## Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform 

### WaveMix Architecture
![image](https://user-images.githubusercontent.com/15833382/201499399-1137bced-61e4-4d3f-9d6c-5b8ba3d43984.png)
### WaveMix-Lite
![image](https://user-images.githubusercontent.com/15833382/172208483-42e6feaa-ff0a-4ccb-8663-f36d25c28d33.png)

To allow image analysis in resource-constrained scenarios without compromising generalizability, we introduce WaveMix -- a novel and flexible neural framework that reduces the GPU RAM (memory) and compute (latency) compared to CNNs and transformers. In addition to using convolutional layers that exploit shift-invariant image statistics, the proposed framework uses multi-level two-dimensional discrete wavelet transform (2D-DWT) modules to exploit scale-invariance and edge sparseness, which gives it the following advantages. Firstly, the fixed weights of wavelet modules do not add to the parameter count while reorganizing information based on these image priors. Secondly, the wavelet modules scale the spatial extents of feature maps by integral powers of 1/2×1/2, which reduces the memory and latency required for forward and backward passes. Finally, a multi-level 2D-DWT leads to a quicker expansion of the receptive field per layer than pooling (which we do not use) and it is a more effective spatial token mixer. WaveMix also generalizes better than other token mixing models, such as ConvMixer, MLP-Mixer, PoolFormer, random filters, and Fourier basis, because the wavelet transform is much better suited for image decomposition and spatial token mixing. WaveMix is a flexible model that can perform well on multiple image tasks without needing architectural modifications. WaveMix achieves a semantic segmentation mIoU of 83% on the Cityscapes validation set outperforming transformer and CNN-based architectures. We also demonstrate the advantages of WaveMix for classification on multiple datasets and show that WaveMix establishes new state-of-the-results in Places-365, EMNIST, and iNAT-mini datasets.


| Task                  | Dataset     | Metric   | Value  |
|-----------------------|-------------|----------|--------|
| Semantic Segmentation | Cityscapes  | mIoU     | 82.70% |
| Image Classification  | ImageNet-1k | Accuracy | 74.93% |

### Parameter Efficiency
| Task                         | Model                                           | Parameters |
|------------------------------|-------------------------------------------------|------------|
| 99% Accu. in MNIST           | WaveMix Lite-8/10                               | 3566       |
| 90% Accu. in Fashion MNIST   | WaveMix Lite-8/5                                | 7156       |
| 80% Accu. in CIFAR-10        | WaveMix Lite-32/7                               | 37058      |
| 90% Accu. in CIFAR-10        | WaveMix Lite-64/6                               | 520106     |   

The high parameter efficiency is obtained by replacing Deconvolution layers with Upsampling

This is an implementation of code from the following papers : [Openreview Paper](https://openreview.net/forum?id=tBoSm4hUWV), [ArXiv Paper 1](https://arxiv.org/abs/2203.03689), [ArXiv Paper 2](https://arxiv.org/abs/2205.14375)



Please cite the following papers if you are using the WaveMix model

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

@misc{https://doi.org/10.48550/arxiv.2205.14375,
  doi = {10.48550/ARXIV.2205.14375},
  
  url = {https://arxiv.org/abs/2205.14375},
  
  author = {Jeevan, Pranav and Viswanathan, Kavitha and S, Anandu A and Sethi, Amit},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.10; I.4.0; I.4.1; I.4.2; I.4.6; I.4.7; I.4.8; I.4.9; I.4.10; I.2.10; I.5.1; I.5.2; I.5.4},
  
  title = {WaveMix: A Resource-efficient Neural Network for Image Analysis},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


```
