import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm

# Custom imports
import models
from models.modules import *
from models.unet import create_diffusion_model
from models.utils import LogMelSpectrogram
from models.discriminator import EnsembleDiscriminator
from training.dataset_vctk import DenoiserDataset
from training.losses import feature_loss, discriminator_loss, generator_loss, melspec_loss
from training.scheduler import ExponentialLRDecay
from training.trainer_vctk import Trainer

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveLLDM: Denoise and Restore your Audio with and End-to-end Diffusion-powered Neural Audio Codec")
    parser.add_argument("--clean_data_path", type=str, required=True, help="Path to clean audio dataset")
    parser.add_argument("--noisy_data_path", type=str, required=True, help="Path to noisy audio dataset")
    parser.add_argument("--add_random_cutting", type=bool, required=True, help="Choose whether random cutting is used or not")
    parser.add_argument("--training_stage", type=int, default=1, help="Define the training stage. Fill 1 if you want to train the autoencoder, fill 2 if you want to fine-tune the autoencoder with noisy data, and fill 3 if you want to train the diffusion model")
    parser.add_argument("--max_cuts", type=int, default=10, help="Define maximum count of random cuts you want")
    parser.add_argument("--cut_duration", type=float, default=0.35, help="Define cut duration for every random cut")
    parser.add_argument("--fixed_length", type=int, default=32678, help="Define number of samples taken if specified")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--use_scheduler", type=bool, default=False, help="Choose whether to use lr scheduler or not")
    return parser.parse_args()

# Initialize models
def create_models(device):
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
        n_codebooks=8,
        levels=[8, 8, 8, 6, 5],
        downsample_factor=[2, 2]
    ).to(device)

    spec_trans = LogMelSpectrogram(
        sample_rate=44100,
        n_mels=160,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        convert_to_fp_16=False
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
    print(f"Loading dataset from {args.clean_data_path} and {args.noisy_data_path}...")

    dataset = DenoiserDataset(
        args.clean_data_path, 
        args.noisy_data_path, 
        args.add_random_cutting, 
        args.training_stage,
        args.max_cuts,
        args.cut_duration,
        args.fixed_length
    )
    
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimizers and scheduler
    gen_opt = torch.optim.Adam(ffgan.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = ExponentialLRDecay(gen_opt)

    # Trainer setup
    trainer = Trainer(
        train_dataloader,
        None,  # Validation dataloader (if any)
        ffgan,
        disc,
        gen_opt,
        disc_opt,
        scheduler if args.use_scheduler else None,
        train_stage = args.training_stage
    )

    print("Starting training...")
    trainer.start_training(epochs=args.epochs)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    train(args)
