from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.lldm_architecture import WaveLLDM
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from training.scheduler import linear_warmup_cosine_decay
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from inspect import isfunction
import gc
import torch
import torch.nn.functional as F
import torch.amp as amp
import os
import torch.distributed as dist

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class WaveLLDMTrainer:
    def __init__(
        self,
        model: WaveLLDM,
        train_dataloader: DataLoader,
        val_dataloader=None,
        optimizer="adamw",
        epochs: int = 300,
        lr: float = 4e-5,
        use_lr_scheduler=True,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 1,
        snapshot_path: str = None,
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_every = save_every
        self.epochs_run = 0
        self.lr = lr

        if optimizer == "adamw":
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optimizer == "adam":
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        if snapshot_path is not None:
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=linear_warmup_cosine_decay())

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.scaler = amp.GradScaler()
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.local_rank == 0 else None

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=f'cuda:{self.local_rank}')
        try:
            self.model.load_state_dict(snapshot["model_state_dict"])
        except:
            self.model.module.load_state_dict(snapshot["model_state_dict"])
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        if self.model.module.use_ema:
            self.model.module.ema.shadow = snapshot["ema_state_dict"]
        self.epochs_run = snapshot["epoch"]
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def train_step(self, batch):
        batch = {k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        self.optimizer.zero_grad()

        with amp.autocast(device_type="cuda"):
            loss, loss_dict = self.model(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected, skipping step")
            self.optimizer.zero_grad()
            return None, None, False

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads):
            print(f"NaN/Inf gradients detected, skipping step")
            self.optimizer.zero_grad()
            return None, None, False

        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.model.module.use_ema:
            self.model.module.ema.update(self.model.module.p_estimator)

        if self.use_lr_scheduler:
            self.scheduler.step()

        return loss, loss_dict, True

    def validate(self, epoch, val_dataloader, writer):
        self.model.module.p_estimator.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, loss_dict = self.model(batch)
                val_loss += loss.item()
                val_steps += 1

                if self.local_rank == 0 and val_steps == 1:
                    self.log_reconstruction(writer, batch, epoch)

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        if self.local_rank == 0:
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        return avg_val_loss

    def train(self):
        for epoch in range(self.epochs_run, self.epochs):
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            with tqdm(total=len(self.train_dataloader), desc=f"[GPU-{self.global_rank}] Epoch {epoch+1}/{self.epochs}", disable=self.local_rank != 0) as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    should_skip = False
                    try:
                        loss, loss_dict, success = self.train_step(batch)
                        if not success:
                            should_skip = True
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            print(f"OOM error on rank {self.global_rank}")
                            torch.cuda.empty_cache()
                            gc.collect()
                            should_skip = True
                        else:
                            raise e
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        should_skip = True

                    skip_tensor = torch.tensor([should_skip], dtype=torch.int, device=self.local_rank)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                    if skip_tensor.item() == 1:
                        self.optimizer.zero_grad()
                        del batch
                        torch.cuda.empty_cache()
                        gc.collect()
                        pbar.update(1)
                        continue

                    train_loss += loss.item()
                    train_steps += 1
                    avg_train_loss = train_loss / train_steps

                    if self.local_rank == 0:
                        pbar.set_postfix({'Loss': avg_train_loss})
                        pbar.update(1)
                        global_step = epoch * len(self.train_dataloader) + idx
                        self.log_to_tensorboard(self.writer, loss_dict, global_step, prefix="train", batch=batch)

                    del loss, loss_dict
                    torch.cuda.empty_cache()

            if self.local_rank == 0:
                avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
                print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {avg_train_loss:.4f}")

                if self.val_dataloader is not None:
                    val_loss = self.validate(epoch, self.val_dataloader, self.writer)
                    print(f"Epoch {epoch+1}/{self.epochs}, Average Val Loss: {val_loss:.4f}")

                if (epoch + 1) % self.save_every == 0:
                    self.save_snapshot(epoch + 1)

        if self.local_rank == 0:
            self.writer.close()

    def log_to_tensorboard(self, writer, loss_dict, global_step, prefix="train", batch=None, num_timesteps=1000):
        for k, v in loss_dict.items():
            writer.add_scalar(f"{prefix}/{k}", v.item(), global_step)
        
        if prefix == "val" and batch is not None:
            with torch.no_grad():
                clean_latents = batch["clean_audio_downsampled_latents"]
                noise = torch.randn_like(clean_latents)
                t = torch.randint(0, num_timesteps, (clean_latents.shape[0],), device=self.local_rank).long()
                z_t = self.model.module.q_sample(x_start=clean_latents, t=t, noise=noise)
                pred = self.model.module.p_estimator(z_t, t, batch["noisy_audio_downsampled_latents"])
                writer.add_histogram(f"{prefix}/pred_histogram", pred.flatten(), global_step)
                writer.add_histogram(f"{prefix}/gt_histogram", clean_latents.flatten(), global_step)

    def log_reconstruction(self, writer, batch, epoch):
        with torch.no_grad():
            degraded_audio_latents = batch["noisy_audio_downsampled_latents"][0].unsqueeze(0)
            recon_audio_latents = self.model.module.sample_with_ema(degraded_audio_latents, batch_size=1)
            
            if self.model.module.std_scale_factor:
                recon_audio_latents *= self.model.module.std_scale_factor

            first_length = batch["lengths"][0]
            degraded_audio_latents = degraded_audio_latents[:, :, :first_length]
            recon_audio_latents = recon_audio_latents[:, :, :first_length]

            recon_audio_latents, _ = self.model.module.quantizer.residual_fsq(recon_audio_latents.mT)
            recon_audio_upsampled_latents = self.model.module.quantizer.upsample(recon_audio_latents.mT)

            diff = batch["melspec_lengths"][0] - recon_audio_upsampled_latents.shape[-1]
            left = max(diff // 2, 0)
            right = max(diff - left, 0)
            
            if diff > 0:
                recon_audio_upsampled_latents = F.pad(recon_audio_upsampled_latents, (left, right))
            elif diff < 0:
                left = max(-diff // 2, 0)
                right = recon_audio_upsampled_latents.shape[-1] + diff + left
                recon_audio_upsampled_latents = recon_audio_upsampled_latents[..., left:right]

            recon_audio = self.model.module.decode(recon_audio_upsampled_latents)
            writer.add_audio("Reconstructed Audio", recon_audio[0].cpu(), epoch, sample_rate=44100)

    def save_snapshot(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.model.module.ema.shadow if self.model.module.use_ema else None
        }
        checkpoint_path = os.path.join(self.save_dir, f"wave_lldm_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
