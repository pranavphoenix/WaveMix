# WaveMix-Lite
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-bymerge)](https://paperswithcode.com/sota/image-classification-on-emnist-bymerge?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-imagenet64x64)](https://paperswithcode.com/sota/image-classification-on-imagenet64x64?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-inat2021-mini)](https://paperswithcode.com/sota/image-classification-on-inat2021-mini?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-byclass)](https://paperswithcode.com/sota/image-classification-on-emnist-byclass?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-places365-standard)](https://paperswithcode.com/sota/image-classification-on-places365-standard?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-letters)](https://paperswithcode.com/sota/image-classification-on-emnist-letters?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-svhn)](https://paperswithcode.com/sota/image-classification-on-svhn?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-stl-10)](https://paperswithcode.com/sota/image-classification-on-stl-10?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=wavemix-resource-efficient-token-mixing-for)

## Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform 

Gains in the ability to generalize on image analysis tasks for neural networks have come at the cost of increased number of parameters and layers, dataset sizes, training and test computations, and GPU RAM. We introduce a new architecture -- WaveMix-Lite -- that can generalize on par with contemporary transformers and convolutional neural networks (CNNs) while needing fewer resources. WaveMix-Lite uses 2D-discrete wavelet transform to efficiently mix spatial information from pixels. WaveMix-Lite seems to be a versatile and scalable architectural framework that can be used for multiple vision tasks, such as image classification and semantic segmentation, without requiring significant architectural changes, unlike transformers and CNNs. It is able to meet or exceed several accuracy benchmarks while training on a single GPU. For instance, it achieves state-of-the-art accuracy on five EMNIST datasets, outperforms CNNs and transformers in ImageNet-1K (64 x 64 images), and achieves an mIoU of 75.32 % on Cityscapes validation set, while using less than one-fifth the number parameters and half the GPU RAM of comparable CNNs or transformers. Our experiments show that while the convolutional elements of neural architectures exploit the properties of shift-invariance and sparseness of edges in images, new types of layers (e.g., wavelet transform) can exploit additional properties of images, such as scale-invariance and finite spatial extents of objects.

### Parameter Efficiency
| Task                         | Model                                           | Parameters |
|------------------------------|-------------------------------------------------|------------|
| 99% Accu. in MNIST           | WaveMix Lite-8/10                               | 3566       |
| 90% Accu. in Fashion MNIST   | WaveMix Lite-8/5                                | 7156       |
| 80% Accu. in CIFAR-10        | WaveMix Lite-32/7                               | 37058      |
| 90% Accu. in CIFAR-10        | WaveMix Lite-64/6                               | 520106     |   

The high parameter efficiency is obtained by replacing Deconvolution layers with Upsampling

This is an implementation of code from [Openreview Paper](https://openreview.net/forum?id=tBoSm4hUWV) and [ArXiv Paper](https://arxiv.org/abs/2203.03689)


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
```
