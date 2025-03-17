import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

import math
from functools import partial
from math import prod
from dataclasses import dataclass
from typing import Callable, Optional

from .modules import *

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            ConvolutionalNet(
                input_channels,
                dims[0],
                kernel_size=7,
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=kernel_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)

class HiFiGANGenerator(nn.Module):
    def __init__(self,
                 *,
                 hop_length: int = 512,
                 upsample_rates: tuple[int] = (8, 8, 2, 2, 2),
                 upsample_kernel_sizes: tuple[int] = (16, 16, 8, 2, 2),
                 resblock_kernel_sizes: tuple[int] = (3, 7, 11),
                 resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 num_mels: int = 128,
                 upsample_initial_channel: int = 512,
                 pre_conv_kernel_size: int = 7,
                 post_conv_kernel_size: int = 7,
                 post_activation: Callable = partial(nn.SiLU, inplace=True)
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = ConvolutionalNet(
            num_mels,
            upsample_initial_channel,
            pre_conv_kernel_size,
            stride = 1
        ).weight_norm()

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                TransConvNet(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i + 1)),
                    k,
                    stride=u
                ).weight_norm()
            )

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                ParallelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = post_activation()

        self.conv_post = ConvolutionalNet(
            ch, 1, post_conv_kernel_size, stride=1
        ).weight_norm()

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)
            x = self.resblocks[i](x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
        
    def remove_parametrizations(self):
        for up in self.ups:
            up.remove_parametrizations()
        for block in self.resblocks:
            block.remove_parametrizations()
        self.conv_pre.remove_parametrizations()
        self.conv_post.remove_parametrizations()

class FireflyArchitecture(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        quantizer: nn.Module,
        spec_transform: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def forward(self, x: torch.Tensor, template=None, mask=None) -> torch.Tensor:
        if self.spec_transform is not None:
            with torch.no_grad():
                x = self.spec_transform(x)

        x = self.backbone(x)
        if mask is not None:
            x = x * mask

        if self.quantizer is not None:
            vq_result = self.quantizer(x)
            x = vq_result.z

            if mask is not None:
                x = x * mask
                
        x = self.head(x)

        if x.ndim == 2:
            x = x[:, None, :]

        return x, vq_result

    def encode(self, audios, audio_lengths):
        audios = audios.float()

        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        # Encode
        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths) -> torch.Tensor:
        mel_masks = sequence_mask(
            feature_lengths * self.downsample_factor,
            indices.shape[2] * self.downsample_factor,
        )
        mel_masks_float_conv = mel_masks[:, None, :].float()
        audio_lengths = (
            feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        )

        audio_masks = sequence_mask(
            audio_lengths,
            indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length,
        )
        audio_masks_float_conv = audio_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv

        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.backbone, "remove_parametrizations"):
            self.backbone.remove_parametrizations()

        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()

    @property
    def device(self):
        return next(self.parameters()).device
