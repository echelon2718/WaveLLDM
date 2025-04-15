import argparse
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import models
from models.modules import *
from models.unet2d import create_diffusion_model
from models.utils import LogMelSpectrogram
from training.dataset_vctk import DenoiserDataset, collate_fn_latents
from models.lldm_architecture import WaveLLDM
from training.trainer_wavelldm import WaveLLDMTrainer

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveLLDM: Denoise and Restore your Audio with an End-to-end Diffusion-powered Neural Audio Codec")
    parser.add_argument("--clean_data_path", type=str, required=True, help="Path to clean audio dataset")
    parser.add_argument("--noisy_data_path", type=str, required=True, help="Path to noisy audio dataset")
    parser.add_argument("--add_random_cutting", type=bool, default=True, help="Choose whether random cutting is used or not")
    parser.add_argument("--max_cuts", type=int, default=10, help="Define maximum count of random cuts")
    parser.add_argument("--cut_duration", type=float, default=0.35, help="Define cut duration for every random cut")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--use_lr_scheduler", type=bool, default=True, help="Choose whether using LR scheduler or not")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--pretrained_codec_path", type=str, default="./pretrained_models/generator_step_142465.pth", help="Path to pretrained codec model")
    parser.add_argument("--snapshot_path", type=str, default=None, help="Path to latest training snapshot (optional)")
    return parser.parse_args()

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    return local_rank

def train(args):
    # Initialize DDP environment
    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    # ----------------------
    # Model components setup
    # ----------------------
    backbone = models.ConvNeXtEncoder(
        input_channels=160,
        depths=[3, 3, 9, 3],
        dims=[128, 256, 384, 512],
        drop_path_rate=0.2,
        kernel_size=7
    ).to(device)

    head = models.HiFiGANGenerator(
        hop_length=512,
        upsample_rates=[8, 8, 2, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        num_mels=512,
        upsample_initial_channel=512,
        pre_conv_kernel_size=13,
        post_conv_kernel_size=13
    ).to(device)

    quantizer = models.DownsampleFSQ(
        input_dim=512,
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
        win_length=2048
    ).to(device)

    ffgan = models.FireflyArchitecture(
        backbone=backbone,
        head=head,
        quantizer=quantizer,
        spec_transform=spec_trans
    ).to(device)

    unet = create_diffusion_model(
        in_channels=2,
        base_channels=32,
        out_channels=1,
        time_dim=32
    ).to(device)

    # Load pretrained weights and set to evaluation mode
    ffgan.load_state_dict(torch.load(args.pretrained_codec_path))
    ffgan.eval()

    # -----------------------------
    # CPU components for preprocessing
    # -----------------------------
    spec_trans_cpu = LogMelSpectrogram(
        sample_rate=44100,
        n_mels=160,
        n_fft=2048,
        hop_length=512,
        win_length=2048
    )

    encoder_cpu = models.ConvNeXtEncoder(
        input_channels=160,
        depths=[3, 3, 9, 3],
        dims=[128, 256, 384, 512],
        drop_path_rate=0.2,
        kernel_size=7
    )

    quantizer_cpu = models.DownsampleFSQ(
        input_dim=512,
        n_groups=8,
        n_codebooks=8,
        levels=[8, 8, 8, 6, 5],
        downsample_factor=[2, 2]
    )

    decoder_cpu = models.HiFiGANGenerator(
        hop_length=512,
        upsample_rates=[8, 8, 2, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        num_mels=512,
        upsample_initial_channel=512,
        pre_conv_kernel_size=13,
        post_conv_kernel_size=13
    )

    # Load pretrained weights into CPU components
    encoder_cpu.load_state_dict(ffgan.backbone.state_dict())
    quantizer_cpu.load_state_dict(ffgan.quantizer.state_dict())
    decoder_cpu.load_state_dict(ffgan.head.state_dict())

    # ----------------------
    # Setup datasets and Distributed Samplers
    # ----------------------
    train_ds = DenoiserDataset(
        args.clean_data_path,
        args.noisy_data_path,
        args.add_random_cutting,
        stage=3,
        spec_trans=spec_trans_cpu,
        encoder=encoder_cpu,
        quantizer=quantizer_cpu,
        device=device
    )

    val_ds = DenoiserDataset(
        args.clean_data_path.replace("train", "test"),
        args.noisy_data_path.replace("train", "test"),
        args.add_random_cutting,
        stage=3,
        spec_trans=spec_trans_cpu,
        encoder=encoder_cpu,
        quantizer=quantizer_cpu,
        device=device
    )

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)

    # DataLoaders
    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_latents,
        pin_memory=True,  # Faster data transfer to GPU
        sampler=train_sampler
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_latents,
        pin_memory=True,
        sampler=val_sampler
    )

    # ----------------------
    # Build and wrap the WaveLLDM model with DDP
    # ----------------------
    wavelldm = WaveLLDM(
        p_estimator=unet,
        learn_logvar=False,
        encoder=ffgan.backbone,
        quantizer=ffgan.quantizer,
        decoder=ffgan.head,
        beta_scheduler="cosine",
        device=device
    )

    # Initialize and train
    trainer = WaveLLDMTrainer(
        model=wavelldm,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        use_lr_scheduler=args.use_lr_scheduler,
        save_dir="./checkpoints",
        log_dir="./logs/wavelldm",
        save_every=5,
        snapshot_path=args.snapshot_path
    )

    print("Starting training 2nd-stage model...")
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    train(args)
