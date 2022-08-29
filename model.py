# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from functools import partial
from typing import Any, List, Union

import torch
from torch import Tensor
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

from utils import make_divisible

__all__ = [
    "EfficientNetV1",
    "efficientnet_v1_b0", "efficientnet_v1_b1", "efficientnet_v1_b2", "efficientnet_v1_b3", "efficientnet_v1_b4",
    "efficientnet_v1_b5", "efficientnet_v1_b6", "efficientnet_v1_b7",
]

efficientnet_cfg = [
    [1, 3, 1, 32, 16, 1],
    [6, 3, 2, 16, 24, 2],
    [6, 5, 2, 24, 40, 2],
    [6, 3, 2, 40, 80, 3],
    [6, 5, 1, 80, 112, 3],
    [6, 5, 2, 112, 192, 4],
    [6, 3, 1, 192, 320, 1],
]


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
            padding: int,
            expand_ratio: float,
            width_mult: float = 1.0,
            stochastic_depth_prob: float = 0.2,
    ) -> None:
        super(MBConv, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        in_channels = make_divisible(in_channels * width_mult, 8, None)
        out_channels = make_divisible(out_channels * width_mult, 8, None)
        expanded_channels = make_divisible(in_channels * expand_ratio, 8, None)

        block: List[nn.Module] = []

        # expand
        if expanded_channels != in_channels:
            block.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                    activation_layer=partial(nn.SiLU, inplace=True),
                    bias=False,
                )
            )

        # depth-wise
        block.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=expanded_channels,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=partial(nn.SiLU, inplace=True),
                bias=False,
            )
        )

        # squeeze and excitation
        block.append(
            SqueezeExcitation(
                expanded_channels,
                max(1, in_channels // 4),
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        block.append(
            Conv2dNormActivation(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=None,
                bias=False,
            )
        )

        self.block = nn.Sequential(*block)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.block(x)
        if self.use_res_connect:
            out = self.stochastic_depth(out)
            out = torch.add(out, identity)

        return out


class EfficientNetV1(nn.Module):

    def __init__(
            self,
            arch_cfg: [Union[int, int, int, int, int, int]],
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            stochastic_depth_prob: float = 0.2,
            dropout: float = 0.2,
            num_classes: int = 1000,
    ) -> None:
        super(EfficientNetV1, self).__init__()

        features: List[nn.Module] = [Conv2dNormActivation(
            3,
            make_divisible(arch_cfg[0][3] * width_mult, 8, None),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
            activation_layer=partial(nn.SiLU, inplace=True),
            bias=False,
        )]

        total_stage_blocks = sum(int(math.ceil(cfg[5] * depth_mult)) for cfg in arch_cfg)
        stage_block_id = 0
        for cfg in arch_cfg:
            stage: List[nn.Module] = []
            for _ in range(int(math.ceil(cfg[5] * depth_mult))):
                # overwrite info if not the first conv in the stage
                if stage:
                    cfg[3] = cfg[4]
                    cfg[2] = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(
                    MBConv(
                        cfg[3],
                        cfg[4],
                        cfg[1],
                        cfg[2],
                        cfg[1] // 2,
                        cfg[0],
                        width_mult,
                        sd_prob,
                    )
                )
                stage_block_id += 1

            features.append(nn.Sequential(*stage))

        # building last several layers
        last_in_channels = make_divisible(arch_cfg[-1][4] * width_mult, 8, None)
        last_out_channels = int(4 * last_in_channels)
        features.append(
            Conv2dNormActivation(
                last_in_channels,
                last_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=partial(nn.SiLU, inplace=True),
                bias=False,
            )
        )

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout, True),
            nn.Linear(last_out_channels, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init_range = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
                nn.init.zeros_(module.bias)


def efficientnet_v1_b0(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.0, 1.0, 0.2, 0.2, **kwargs)

    return model


def efficientnet_v1_b1(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.0, 1.1, 0.2, 0.2, **kwargs)

    return model


def efficientnet_v1_b2(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.1, 1.2, 0.2, 0.3, **kwargs)

    return model


def efficientnet_v1_b3(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.2, 1.4, 0.2, 0.3, **kwargs)

    return model


def efficientnet_v1_b4(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.4, 1.8, 0.2, 0.4, **kwargs)

    return model


def efficientnet_v1_b5(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.6, 2.2, 0.2, 0.4, **kwargs)

    return model


def efficientnet_v1_b6(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 1.8, 2.6, 0.2, 0.5, **kwargs)

    return model


def efficientnet_v1_b7(**kwargs: Any) -> EfficientNetV1:
    model = EfficientNetV1(efficientnet_cfg, 2.0, 3.1, 0.2, 0.5, **kwargs)

    return model
