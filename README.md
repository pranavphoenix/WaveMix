# WaveMix 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-bymerge)](https://paperswithcode.com/sota/image-classification-on-emnist-bymerge?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-imagenet64x64)](https://paperswithcode.com/sota/image-classification-on-imagenet64x64?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-inat2021-mini)](https://paperswithcode.com/sota/image-classification-on-inat2021-mini?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-byclass)](https://paperswithcode.com/sota/image-classification-on-emnist-byclass?p=wavemix-resource-efficient-token-mixing-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wavemix-resource-efficient-token-mixing-for/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=wavemix-resource-efficient-token-mixing-for)

## Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform 

Although certain vision transformer (ViT) and CNN architectures generalize well on vision tasks, it is often impractical to use them on green, edge, or desktop computing due to their computational requirements for training and even testing. We present WaveMix as an alternative neural architecture that uses a multi-scale 2D discrete wavelet transform (DWT) for spatial token mixing. Unlike ViTs, WaveMix neither unrolls the image nor requires self-attention of quadratic complexity. Additionally, DWT introduces another inductive bias -- besides convolutional filtering -- to utilize the 2D structure of an image to improve generalization. The multi-scale nature of the DWT also reduces the requirement for a deeper architecture compared to the CNNs, as the latter relies on pooling for partial spatial mixing. WaveMix models show generalization that is competitive with ViTs, CNNs, and token mixers on several datasets while requiring lower GPU RAM (training and testing), number of computations, and storage.

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
