# WaveMix-Lite
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-byclass)](https://paperswithcode.com/sota/image-classification-on-emnist-byclass?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural-1/scene-classification-on-places365-standard)](https://paperswithcode.com/sota/scene-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=wavemix-lite-a-resource-efficient-neural) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-letters)](https://paperswithcode.com/sota/image-classification-on-emnist-letters?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-emnist-bymerge)](https://paperswithcode.com/sota/image-classification-on-emnist-bymerge?p=wavemix-lite-a-resource-efficient-neural) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-imagenet64x64)](https://paperswithcode.com/sota/image-classification-on-imagenet64x64?p=wavemix-lite-a-resource-efficient-neural) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-inat2021-mini)](https://paperswithcode.com/sota/image-classification-on-inat2021-mini?p=wavemix-lite-a-resource-efficient-neural) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-caltech-256)](https://paperswithcode.com/sota/image-classification-on-caltech-256?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural-1/image-classification-on-places365-standard)](https://paperswithcode.com/sota/image-classification-on-places365-standard?p=wavemix-lite-a-resource-efficient-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=wavemix-lite-a-resource-efficient-neural) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=wavemix-lite-a-resource-efficient-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural/image-classification-on-svhn)](https://paperswithcode.com/sota/image-classification-on-svhn?p=wavemix-lite-a-resource-efficient-neural)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural-1/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=wavemix-lite-a-resource-efficient-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-lite-a-resource-efficient-neural-1/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=wavemix-lite-a-resource-efficient-neural-1)

## Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform 

![image](https://user-images.githubusercontent.com/15833382/172208483-42e6feaa-ff0a-4ccb-8663-f36d25c28d33.png)

Gains in the ability to generalize on image analysis tasks for neural networks have come at the cost of increased number of parameters and layers, dataset sizes, training and test computations, and GPU RAM. We introduce a new architecture -- WaveMix-Lite -- that can generalize on par with contemporary transformers and convolutional neural networks (CNNs) while needing fewer resources. WaveMix-Lite uses 2D-discrete wavelet transform to efficiently mix spatial information from pixels. WaveMix-Lite seems to be a versatile and scalable architectural framework that can be used for multiple vision tasks, such as image classification and semantic segmentation, without requiring significant architectural changes, unlike transformers and CNNs. It is able to meet or exceed several accuracy benchmarks while training on a single GPU. For instance, it achieves state-of-the-art accuracy on five EMNIST datasets, outperforms CNNs and transformers in ImageNet-1K (64 x 64 images), and achieves an mIoU of 75.32 % on Cityscapes validation set, while using less than one-fifth the number parameters and half the GPU RAM of comparable CNNs or transformers. Our experiments show that while the convolutional elements of neural architectures exploit the properties of shift-invariance and sparseness of edges in images, new types of layers (e.g., wavelet transform) can exploit additional properties of images, such as scale-invariance and finite spatial extents of objects.

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
  author = {Jeevan, Pranav and Viswanathan, Kavitha and Sethi, Amit},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.10; I.4.0; I.4.1; I.4.2; I.4.6; I.4.7; I.4.8; I.4.9; I.4.10; I.2.10; I.5.1; I.5.2; I.5.4},
  title = {WaveMix-Lite: A Resource-efficient Neural Network for Image Analysis},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
