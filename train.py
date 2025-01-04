import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm

import models
from models.modules import *
from models.unet import create_diffusion_model
from models.utils import LogMelSpectrogram
from models.discriminator import EnsembleDiscriminator
from training.dataset import MyDataset
from training.losses import feature_loss, discriminator_loss, generator_loss, melspec_loss
from training.scheduler import CycleScheduler
from training.trainer import Trainer
from datasets import load_from_disk

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion-based TTS model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    return parser.parse_args()

# Initialize models
def create_models(device):
    model = create_diffusion_model(
        in_channels=8,
        base_channels=16,
        out_channels=8
    ).to(device)

    backbone = models.ConvNeXtEncoder(
        input_channels=160,
        depths=[3, 3, 9, 3],
        dims=[128, 256, 288, 384],
        drop_path_rate=0.2,
        kernel_size=7
    ).to(device)

    head = models.HiFiGANGenerator(
        hop_length=512,
        upsample_rates=[8, 8, 2, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        num_mels=384,
        upsample_initial_channel=384,
        pre_conv_kernel_size=13,
        post_conv_kernel_size=13
    ).to(device)

    quantizer = models.DownsampleFSQ(
        input_dim=384,
        n_groups=8,
        n_codebooks=1,
        levels=[8, 5, 5, 5],
        downsample_factor=[2, 2]
    ).to(device)

    spec_trans = LogMelSpectrogram(
        sample_rate=44100,
        n_mels=160,
        n_fft=2048,
        hop_length=512,
        win_length=2048
    ).to(device)

    ffgan = models.FireflyArchitecture(
        backbone=backbone,
        head=head,
        quantizer=quantizer,
        spec_transform=spec_trans
    ).to(device)

    disc = EnsembleDiscriminator(periods=[2, 3, 5, 7, 11, 17, 23, 37]).to(device)

    return ffgan, disc

# Training function
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ffgan, disc = create_models(device)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    ds = load_from_disk(args.data_path)
    dataset = MyDataset(ds)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimizers and scheduler
    gen_opt = torch.optim.Adam(ffgan.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    cyc_sched = CycleScheduler(gen_opt, args.lr, n_iter=len(train_dataloader) * args.epochs, momentum=None, warmup_proportion=0.05)

    # Trainer setup
    trainer = Trainer(
        train_dataloader,
        None,  # Validation dataloader (if any)
        ffgan,
        disc,
        gen_opt,
        disc_opt,
        cyc_sched
    )

    print("Starting training...")
    trainer.start_training(epochs=args.epochs)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    train(args)
