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
![image](https://user-images.githubusercontent.com/15833382/226090639-b4571494-7d2d-4bcb-81e3-127916339dfe.png)

### WaveMix-Lite
![image](https://user-images.githubusercontent.com/15833382/226090664-d844e4f1-854a-43b3-8106-78307f187fe8.png)

We propose WaveMix– a novel neural architecture for computer vision that is resource-efficient yet generalizable and scalable. WaveMix networks achieve comparable or better accuracy than the state-of-the-art convolutional neural networks, vision transformers, and token mixers for several tasks, establishing new benchmarks for segmentation on Cityscapes; and for classification on Places-365, f ive EMNIST datasets, and iNAT-mini. Remarkably, WaveMix architectures require fewer parameters to achieve these benchmarks compared to the previous state-of-the-art. Moreover, when controlled for the number of parameters, WaveMix requires lesser GPU RAM, which translates to savings in time, cost, and energy. To achieve these gains we used multi-level two-dimensional discrete wavelet transform (2D-DWT) in WaveMix blocks, which has the following advantages: (1) It reorganizes spatial information based on three strong image priors– scale-invariance, shift-invariance, and sparseness of edges, (2) in a lossless manner without adding parameters, (3) while also reducing the spatial sizes of feature maps, which reduces the memory and time required for forward and backward passes, and (4) expanding the receptive field faster than convolutions do. The whole architecture is a stack of self-similar and resolution-preserving WaveMix blocks, which allows architectural f lexibility for various tasks and levels of resource availability.


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

@misc{jeevan2023wavemix,
      title={WaveMix: A Resource-efficient Neural Network for Image Analysis}, 
      author={Pranav Jeevan and Kavitha Viswanathan and Anandu A S and Amit Sethi},
      year={2023},
      eprint={2205.14375},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
