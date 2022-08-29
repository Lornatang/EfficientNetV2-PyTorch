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
    "EfficientNetV2",
    "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
]

efficientnet_v2_s_cfg = [
    ["FusedMBConv", 1, 3, 1, 24, 24, 2],
    ["FusedMBConv", 4, 3, 2, 24, 48, 4],
    ["FusedMBConv", 4, 3, 2, 48, 64, 4],
    ["MBConv", 4, 3, 2, 64, 128, 6],
    ["MBConv", 6, 3, 1, 128, 160, 9],
    ["MBConv", 6, 3, 2, 160, 256, 15],
]

efficientnet_v2_m_cfg = [
    ["FusedMBConv", 1, 3, 1, 24, 24, 3],
    ["FusedMBConv", 4, 3, 2, 24, 48, 5],
    ["FusedMBConv", 4, 3, 2, 48, 80, 5],
    ["MBConv", 4, 3, 2, 80, 160, 7],
    ["MBConv", 6, 3, 1, 160, 176, 14],
    ["MBConv", 6, 3, 2, 176, 304, 18],
    ["MBConv", 6, 3, 1, 304, 512, 5],
]

efficientnet_v2_l_cfg = [
    ["FusedMBConv", 1, 3, 1, 32, 32, 4],
    ["FusedMBConv", 4, 3, 2, 32, 64, 7],
    ["FusedMBConv", 4, 3, 2, 64, 96, 7],
    ["MBConv", 4, 3, 2, 96, 192, 10],
    ["MBConv", 6, 3, 1, 192, 224, 19],
    ["MBConv", 6, 3, 2, 224, 384, 25],
    ["MBConv", 6, 3, 1, 384, 640, 7],
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
            stochastic_depth_prob: float = 0.2,
    ) -> None:
        super(MBConv, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
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
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001),
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
                norm_layer=partial(nn.BatchNorm2d, eps=0.001),
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
                norm_layer=partial(nn.BatchNorm2d, eps=0.001),
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


class FusedMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
            padding: int,
            expand_ratio: float,
            stochastic_depth_prob: float = 0.2,
    ) -> None:
        super(FusedMBConv, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded_channels = make_divisible(in_channels * expand_ratio, 8, None)

        block: List[nn.Module] = []

        # expand
        if expanded_channels != in_channels:
            # fused expand
            block.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    groups=1,
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001),
                    activation_layer=partial(nn.SiLU, inplace=True),
                    bias=False,
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
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001),
                    activation_layer=None,
                    bias=False,
                )
            )
        else:
            block.append(
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    groups=1,
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001),
                    activation_layer=partial(nn.SiLU, inplace=True),
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


class EfficientNetV2(nn.Module):

    def __init__(
            self,
            arch_cfg: [Union[str, int, int, int, int, int, int]],
            stochastic_depth_prob: float = 0.2,
            dropout: float = 0.2,
            num_classes: int = 1000,
    ) -> None:
        super(EfficientNetV2, self).__init__()

        features: List[nn.Module] = [Conv2dNormActivation(
            3,
            arch_cfg[0][4],
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001),
            activation_layer=partial(nn.SiLU, inplace=True),
            bias=False,
        )]

        total_stage_blocks = sum(int(math.ceil(cfg[6])) for cfg in arch_cfg)
        stage_block_id = 0
        for cfg in arch_cfg:
            stage: List[nn.Module] = []
            for _ in range(int(math.ceil(cfg[6]))):
                # Chose feature block
                if cfg[0] == "MBConv":
                    block = MBConv
                else:
                    block = FusedMBConv
                # overwrite info if not the first conv in the stage
                if stage:
                    cfg[4] = cfg[5]
                    cfg[3] = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(
                    block(
                        cfg[4],
                        cfg[5],
                        cfg[2],
                        cfg[3],
                        cfg[2] // 2,
                        cfg[1],
                        sd_prob,
                    )
                )
                stage_block_id += 1

            features.append(nn.Sequential(*stage))

        # building last several layers
        last_in_channels = make_divisible(arch_cfg[-1][5], 8, None)
        last_out_channels = 1280
        features.append(
            Conv2dNormActivation(
                last_in_channels,
                last_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001),
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


def efficientnet_v2_s(**kwargs: Any) -> EfficientNetV2:
    model = EfficientNetV2(efficientnet_v2_s_cfg, 0.2, 0.2, **kwargs)

    return model


def efficientnet_v2_m(**kwargs: Any) -> EfficientNetV2:
    model = EfficientNetV2(efficientnet_v2_m_cfg, 0.2, 0.3, **kwargs)

    return model


def efficientnet_v2_l(**kwargs: Any) -> EfficientNetV2:
    model = EfficientNetV2(efficientnet_v2_l_cfg, 0.2, 0.4, **kwargs)

    return model
