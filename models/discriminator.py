import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from .modules import get_padding

class MultiperiodDisc(nn.Module):
    def __init__(
        self,
        period,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        channels: list = [1, 32, 128, 512, 1024, 1024]
    ):
        super(MultiperiodDisc, self).__init__()
        self.period = period
        normalize = weight_norm if use_spectral_norm == False else spectral_norm

        self.conv = nn.ModuleList(
            [normalize(
                nn.Conv2d(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    kernel_size=(kernel_size, 1), 
                    stride=(stride,1),
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ) for i in range(len(channels) - 1)]
        )

        self.post_conv = normalize(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.conv:
            x = l(x)
            x = F.leaky_relu(x, 0.1, inplace=False)
            fmap.append(x)
        x = self.post_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiscaleDisc(nn.Module):
    def __init__(
        self, 
        use_spectral_norm : bool = False,
        channels : list  = [1, 128, 128, 256, 512, 1024, 1024, 1024],
        kernels : list   = [15, 41, 41, 41, 41, 41, 5],
        strides : list   = [1, 2, 2, 4, 4, 1, 1],
        group_lvl : list = [1, 4, 16, 16, 16, 16, 1],
        paddings : list  = [7, 20, 20, 20, 20, 20, 2]
    ):
        super(MultiscaleDisc, self).__init__()
        normalize = weight_norm if use_spectral_norm == False else spectral_norm
        self.conv = nn.ModuleList(
            [
                normalize(
                    nn.Conv1d(
                        in_channels = channels[i],
                        out_channels = channels[i+1],
                        kernel_size = kernels[i],
                        stride = strides[i],
                        padding = paddings[i],
                        groups = group_lvl[i]
                    )
                ) for i in range(len(channels) - 1)
            ]
        )
        
        self.post_conv = normalize(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.conv:
            x = l(x)
            x = F.leaky_relu(x, 0.1, inplace=False)
            fmap.append(x)
        x = self.post_conv(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class EnsembleDiscriminator(nn.Module):
    def __init__(self, ckpt_path=None, periods=(2, 3, 5, 7, 11)):
        super(EnsembleDiscriminator, self).__init__()

        discs = [MultiscaleDisc(use_spectral_norm=True)]
        discs = discs + [MultiperiodDisc(i, use_spectral_norm=False) for i in periods]
        self.discriminators = nn.ModuleList(discs)

        if ckpt_path is not None:
            self.restore_from_ckpt(ckpt_path)

    def restore_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mpd, msd = ckpt["mpd"], ckpt["msd"]

        all_keys = {}
        for k, v in mpd.items():
            keys = k.split(".")
            keys[1] = str(int(keys[1]) + 1)
            all_keys[".".join(keys)] = v

        for k, v in msd.items():
            if not k.startswith("discriminators.0"):
                continue
            all_keys[k] = v

        self.load_state_dict(all_keys, strict=True)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
