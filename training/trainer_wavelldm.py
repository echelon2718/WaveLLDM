from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.lldm_architecture import WaveLLDM
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from training.scheduler import linear_warmup_cosine_decay
from torch.optim.lr_scheduler import LambdaLR
from inspect import isfunction
import gc

import torch
import torch.nn.functional as F
import torch.cuda.amp as amp  # For mixed precision training
import os

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# Modified WaveLLDMTrainer
class WaveLLDMTrainer:
    def __init__(
        self,
        model: WaveLLDM,
        train_dataloader: DataLoader,
        val_dataloader=None,
        optimizer="adamw",
        epochs: int = 300,
        lr: float = 4e-5,
        use_lr_scheduler = True,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 5,
        snapshot_path: str = None,
        device: str = "cuda"
    ):
        self.device = device
        self.model = model.to(device)
        self.model.encoder = self.model.encoder.to(device)
        self.model.decoder = self.model.decoder.to(device)
        self.model.quantizer = self.model.quantizer.to(device)
        self.model.p_estimator = self.model.p_estimator.to(device)
        self.model.device = device
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
            print("Loading snapshot...")
            self._load_snapshot(snapshot_path)

        self.use_lr_scheduler = use_lr_scheduler

        # Scheduler: define total training steps and warmup
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            total_steps = epochs * 1925
            warmup_steps = int(0.1 * total_steps)
            # LambdaLR expects a function mapping current step to a multiplier
            lr_lambda = linear_warmup_cosine_decay(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            # step() must be called per batch
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            self.scheduler = None

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.scaler = amp.GradScaler()
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        try:
            self.model.load_state_dict(snapshot["model_state_dict"])
        except:
            # Handle the case where the model is wrapped in DDP
            self.model.module.load_state_dict(snapshot["model_state_dict"])

        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        try:
            self.model.ema.shadow = snapshot["ema_state_dict"]
        except:
            # Handle the case where the model is wrapped in DDP
            self.model.module.ema.shadow = snapshot["ema_state_dict"]
        self.epochs_run = snapshot["epoch"]
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def train_step(self, batch):
        batch["clean_audios"] = batch["clean_audios"].to(self.device)
        batch["noisy_audios"] = batch["noisy_audios"].to(self.device)

        self.optimizer.zero_grad()

        loss, loss_dict = self.model(batch)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_lr_scheduler:
            self.scheduler.step()

        if self.model.use_ema:
            self.model.ema.update(self.model.p_estimator)
        
        return loss, loss_dict
    
    def validate(self, epoch, val_dataloader, writer):
        self.model.p_estimator.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch["clean_audios"] = batch["clean_audios"].to(self.device)
                batch["noisy_audios"] = batch["noisy_audios"].to(self.device)

                loss, loss_dict = self.model(batch)
                val_loss += loss.item()
                val_steps += 1

                self.log_to_tensorboard(writer, loss_dict, val_steps + epoch * len(val_dataloader), prefix="val", batch=batch)

                if val_steps == 1:
                    self.log_reconstruction(writer, batch, epoch)
        
        avg_val_loss = val_loss / val_steps
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        return avg_val_loss
    

    def train(self):
        for epoch in range(self.epochs_run, self.epochs):
            self.model.p_estimator.train()
            train_loss = 0.0
            train_steps = 0

            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    try:
                        self.optimizer.zero_grad()
                        loss, loss_dict = self.train_step(batch)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"NaN/Inf loss at step {idx}, skipping batch")
                            continue

                        train_loss += loss.item()
                        train_steps += 1
                        avg_train_loss = train_loss / train_steps

                        pbar.set_postfix({'Loss': avg_train_loss})
                        pbar.update(1)

                        if idx % 100 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()

                        global_step = epoch * len(self.train_dataloader) + train_steps
                        self.log_to_tensorboard(self.writer, loss_dict, global_step, prefix="train", batch=batch)

                    except RuntimeError as e:
                        print(f"Error at step {idx}: {e}")
                        torch.cuda.empty_cache()
                        continue

            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {avg_train_loss:.4f}")

            if self.val_dataloader is not None:
                val_loss = self.validate(epoch, self.val_dataloader, self.writer)
                print(f"Epoch {epoch+1}/{self.epochs}, Average Val Loss: {val_loss:.4f}")

            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                self.save_snapshot(epoch + 1)
                print(f"Saved checkpoint to {checkpoint_path}")

        self.writer.close()
    
    def log_to_tensorboard(self, writer, loss_dict, global_step, prefix="train", batch=None, noise=None, num_timesteps=1000):
        for k, v in loss_dict.items():
            writer.add_scalar(f"{k}", v.item(), global_step)
        
        if prefix == "val":
            assert batch is not None, "Batch must be provided for validation logging"
            with torch.no_grad():
                # Dapatkan audio
                clean_audios = batch["clean_audios"] # Shape: B, 1, L
                noisy_audios = batch["noisy_audios"] # Shape: B, 1, L
    
                clean_spec = self.model.spec_trans(clean_audios) # Shape: B, n_mel, l
                noisy_spec = self.model.spec_trans(noisy_audios) # Shape: B, n_mel, l

                clean_latents = self.model.encode(clean_spec) # Shape: B, base_dim, l_latent
                noisy_latents = self.model.encode(noisy_spec) # Shape: B, base_dim, l_latent

                noise = default(noise, lambda: torch.randn_like(clean_latents))
                t = torch.randint(0, num_timesteps, (clean_latents.shape[0],), device=self.device).long()
                z_t = self.model.q_sample(x_start=clean_latents, t=t, noise=noise)
                pred = self.model.p_estimator(z_t, t, noisy_latents)
                writer.add_histogram(f"{prefix}/pred_histogram", pred.flatten(), global_step)
                writer.add_histogram(f"{prefix}/gt_histogram", clean_latents.flatten(), global_step)
    
    def log_reconstruction(self, writer, batch, epoch):
        with torch.no_grad():
            # Dapatkan audio
            clean_audios = batch["clean_audios"] # Shape: B, 1, L
            noisy_audios = batch["noisy_audios"] # Shape: B, 1, L

            clean_spec = self.model.spec_trans(clean_audios) # Shape: B, n_mel, l
            noisy_spec = self.model.spec_trans(noisy_audios) # Shape: B, n_mel, l
    
            clean_latents = self.model.encode(clean_spec) # Shape: B, base_dim, l_latent
            noisy_latents = self.model.encode(noisy_spec) # Shape: B, base_dim, l_latent

            sample_index = torch.randint(0, clean_latents.shape[0], ()).item()
            
            clean_audio_latents = clean_latents[sample_index].unsqueeze(0)
            degraded_audio_latents = noisy_latents[sample_index].unsqueeze(0)
            

            # 2. Sampel latents dari p_sample_loop
            recon_audio_latents = self.model.sample_with_ema(degraded_audio_latents, batch_size=1)
            
            # Rescale latents if std_scale_factor is provided (1/std_scale_factor * latents)
            if self.model.scaling_factor:
                # recon_audio_latents = (2 * recon_audio_latents - 1) * self.model.scaling_factor
                recon_audio_latents = self.model.scaling_factor * recon_audio_latents

            # 3. Dapatkan informasi mengenai panjang laten asli dari batch
            first_length = clean_latents.shape[-1]

            # 4. Ambil latents yang telah disampel dan dari batch dan hilangkan padding, sehingga kita mendapatkan panjang
            # aslinya.
            degraded_audio_latents = degraded_audio_latents[:, :, :first_length]
            recon_audio_latents = recon_audio_latents[:, :, :first_length]
            clean_audio_latents = clean_audio_latents[:, :, :first_length]

            recon_audio_latents, _ = self.model.quantizer.residual_fsq(recon_audio_latents.mT)
            clean_audio_latents, _ = self.model.quantizer.residual_fsq(clean_audio_latents.mT)

            recon_audio_upsampled_latents = self.model.quantizer.upsample(recon_audio_latents.mT)
            clean_audio_upsampled_latents = self.model.quantizer.upsample(clean_audio_latents.mT)

            diff = clean_spec.shape[-1] - recon_audio_upsampled_latents.shape[-1] # Masih salah, wip
            left = max(diff // 2, 0)  # Pastikan left tidak negatif
            right = max(diff - left, 0)
            
            if diff > 0:
                recon_audio_upsampled_latents = F.pad(recon_audio_upsampled_latents, (left, right))
                clean_audio_upsampled_latents = F.pad(clean_audio_upsampled_latents, (left, right))
            elif diff < 0:
                left = max(-diff // 2, 0)  # Pastikan slicing dimulai dari 0
                right = recon_audio_upsampled_latents.shape[-1] + diff + left
                recon_audio_upsampled_latents = recon_audio_upsampled_latents[..., left:right]
                clean_audio_upsampled_latents = clean_audio_upsampled_latents[..., left:right]

            recon_audio = self.model.decode(recon_audio_upsampled_latents)
            clean_audio = self.model.decode(clean_audio_upsampled_latents)

            writer.add_audio(f"Reconstructed Audio", recon_audio[0].cpu(), epoch, sample_rate=48000)
            writer.add_audio(f"Clean Audio", clean_audio[0].cpu(), epoch, sample_rate=48000)
    
    def save_snapshot(self, epoch):
        """Simpan weight model"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.model.ema.shadow if self.model.use_ema else None
        }
        checkpoint_path = os.path.join(self.save_dir, f"wave_lldm_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"EPOCH {epoch} | Checkpoint saved at {checkpoint_path}")
