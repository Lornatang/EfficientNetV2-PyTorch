# EfficientNetV1-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946v5.pdf).

## Table of contents

- [EfficientNetV1-PyTorch](#efficientnetv1-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](#efficientnet-rethinking-model-scaling-for-convolutional-neural-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `efficientnet_v1_b0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 91: `model_weights_path` change to `./results/pretrained_models/efficientnet_v1_b0-ImageNet_1K-54492891.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `efficientnet_v1_b0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 51: `pretrained_model_weights_path` change to `./results/pretrained_models/efficientnet_v1_b0-ImageNet_1K-54492891.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `efficientnet_v1_b0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 54: `resume` change to `./samples/efficientnet_v1_b0-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1905.11946v5.pdf](https://arxiv.org/pdf/1905.11946v5.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model        |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------------:|:-----------:|:-----------------:|:-----------------:|
| efficientnet_v1_b0 | ImageNet_1K | 22.9%(**26.1%**)  |  6.7%(**8.4%**)   |
| efficientnet_v1_b1 | ImageNet_1K | 20.9%(**21.3%**)  |  5.6%(**5.7%**)   |
| efficientnet_v1_b2 | ImageNet_1K | 19.9%(**22.1%**)  |  5.1%(**6.4%**)   |
| efficientnet_v1_b3 | ImageNet_1K | 18.4%(**18.9%**)  |  4.3%(**4.3%**)   |
| efficientnet_v1_b4 | ImageNet_1K | 17.1%(**16.9%**)  |  3.6%(**3.5%**)   |
| efficientnet_v1_b5 | ImageNet_1K | 16.4%(**16.4%**)  |  3.3%(**3.3%**)   |
| efficientnet_v1_b6 | ImageNet_1K | 16.0%(**16.0%**)  |  3.2%(**3.2%**)   |
| efficientnet_v1_b7 | ImageNet_1K | 15.7%(**15.7%**)  |  3.0%(**3.1%**)   |

```bash
# Download `efficientnet_v1_b0-ImageNet_1K-54492891.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `efficientnet_v1_b0` model successfully.
Load `efficientnet_v1_b0` model weights `/EfficientNetV1-PyTorch/results/pretrained_models/efficientnet_v1_b0-ImageNet_1K-54492891.pth.tar` successfully.
tench, Tinca tinca                                                          (95.10%)
barracouta, snoek                                                           (2.01%)
reel                                                                        (0.10%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.06%)
armadillo                                                                   (0.04%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

*Mingxing Tan, Quoc V. Le*

##### Abstract

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for
better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that
carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we
propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly
effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a
family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In
particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and
6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve
state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order
of magnitude fewer parameters. Source code is
at [this https URL](https//github.com/tensorflow/tpu/tree/master/models/official/efficientnet.)

[[Paper]](https://arxiv.org/pdf/1905.11946v5.pdf)

```bibtex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```