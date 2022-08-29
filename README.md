# EfficientNetV2-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298v3.pdf).

## Table of contents

- [EfficientNetV2-PyTorch](#efficientnetv2-pytorch)
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
        - [EfficientNetV2: Smaller Models and Faster Training](#efficientnetv2-smaller-models-and-faster-training)

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

- line 29: `model_arch_name` change to `efficientnet_v2_s`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 91: `model_weights_path` change to `./results/pretrained_models/EfficientNetV2_S-ImageNet_1K-a93bc34c.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `efficientnet_v2_s`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 51: `pretrained_model_weights_path` change to `./results/pretrained_models/EfficientNetV2_S-ImageNet_1K-a93bc34c.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `efficientnet_v2_s`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 54: `resume` change to `./samples/efficientnet_v2_s-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/2104.00298v3.pdf](https://arxiv.org/pdf/2104.00298v3.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model       |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:-----------------:|:-----------:|:-----------------:|:-----------------:|
| efficientnet_v2_s | ImageNet_1K | 16.1%(**15.7%**)  |    -(**3.1%**)    |
| efficientnet_v2_m | ImageNet_1K | 14.9%(**14.9%**)  |    -(**2.8%**)    |
| efficientnet_v2_l | ImageNet_1K | 114.2%(**14.2%**) |    -(**2.2%**)    |


```bash
# Download `EfficientNetV2_S-ImageNet_1K-a93bc34c.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `efficientnet_v2_s` model successfully.
Load `efficientnet_v2_s` model weights `/EfficientNetV2-PyTorch/results/pretrained_models/EfficientNetV2_S-ImageNet_1K-a93bc34c.pth.tar` successfully.
tench, Tinca tinca                                                          (79.91%)
barracouta, snoek                                                           (0.65%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.14%)
sturgeon                                                                    (0.14%)
coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch    (0.11%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### EfficientNetV2: Smaller Models and Faster Training

*Mingxing Tan, Quoc V. Le*

##### Abstract

This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better
parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware
neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were
searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2
models train much faster than state-of-the-art models while being up to 6.8x smaller.
Our training can be further sped up by progressively increasing the image size during training, but it often causes a
drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout
and data augmentation) as well, such that we can achieve both fast training and good accuracy.
With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and
CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on
ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing
resources. Code will be available at [this https URL](https://github.com/google/automl/tree/master/efficientnetv2).

[[Paper]](https://arxiv.org/pdf/2104.00298v3.pdf)

```bibtex
@inproceedings{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={10096--10106},
  year={2021},
  organization={PMLR}
}
```