import torch
import torch.nn as nn
from model.vae.modules import Encoder, Decoder
from model.vae.distributions import DiagonalGaussianDistribution

from model.components.generator import get_vocoder, vocoder_infer

class AutoencoderKL(nn.Module):
    def __init__(
            self,
            ddconfig = None,
            lossconfig = None,
            image_key = "fbank",
            embed_dim = None,
            time_shuffle = 1,
            subband = 1,
            ckpt_path = None,
            reload_from_ckpt = None,
            ignore_keys = [],
            colorize_nlabels = None,
            monitor = None,
            base_lr = 1e-5
    ):
        super(AutoencoderKL, self).__init__()
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)
        
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.vocoder = get_vocoder(None, "cpu") # Perlu diperhatikan
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        
        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None
    
    def encode(self, x):
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = self.freq_merge_subband(x)
        return x

    def decode_to_waveform(self, dec):
        dec = dec.squeeze(1).permute(2, 0, 1)
        wav_recon = vocoder_infer(dec, self.vocoder)
        return wav_recon
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        if self.flag_first_run:
            print("latent shape: ", z.shape)
            self.flag_first_run = False
        
        x = self.decode(z)

        return x, posterior
    
    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank
        
        B, C, T, F_BIN = fbank.shape

        assert fbank.size(-1) % self.subband == 0
        assert C == 1

        return (
            fbank.squeeze(1)
            .reshape(B, T, self.subband, F_BIN // self.subband)
            .permute(0, 2, 1, 3)
        )
    
    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        
        assert subband_fbank.size(1) == self.subband
        B, C, T, F_BIN = subband_fbank.shape
        return subband_fbank.permute(0, 2, 1, 3).reshape(B, T, -1).unsqueeze(1)
