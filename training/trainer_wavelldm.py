from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.lldm_architecture import WaveLLDM
import gc

import torch
import torch.cuda.amp as amp  # For mixed precision training
import os

# Modified WaveLLDMTrainer
class WaveLLDMTrainer:
    def __init__(
        self,
        model: WaveLLDM,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 300,
        lr: float = 3e-5,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_every = save_every
        self.device = device

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        self.scaler = amp.GradScaler()
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    try:
                        self.optimizer.zero_grad()
                        with amp.autocast():
                            loss, loss_dict = self.model.train_step(batch)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"NaN/Inf loss at step {idx}, skipping batch")
                            continue

                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        train_loss += loss.item()
                        train_steps += 1
                        avg_train_loss = train_loss / train_steps

                        pbar.set_postfix({'Loss': avg_train_loss})
                        pbar.update(1)

                        if idx % 100 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()

                        global_step = epoch * len(self.train_dataloader) + idx
                        self.model.log_to_tensorboard(self.writer, loss_dict, global_step, prefix="train", batch=batch)

                    except RuntimeError as e:
                        print(f"Error at step {idx}: {e}")
                        torch.cuda.empty_cache()
                        continue

            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {avg_train_loss:.4f}")

            if self.val_dataloader is not None:
                val_loss = self.model.validate(epoch, self.val_dataloader, self.writer)
                print(f"Epoch {epoch+1}/{self.epochs}, Average Val Loss: {val_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                self.model.save_checkpoint(epoch + 1)
                print(f"Saved checkpoint to {checkpoint_path}")

        self.writer.close()

### LEGACY VERSION ###
# class WaveLLDMTrainer:
#     def __init__(
#         self,
#         model: WaveLLDM,
#         train_dataloader,
#         val_dataloader=None,
#         epochs: int = 300,
#         save_dir: str = "./checkpoints",
#         log_dir: str = "./logs",
#         save_every: int = 5,
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     ):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.epochs = epochs
#         self.save_dir = save_dir
#         self.log_dir = log_dir
#         self.save_every = save_every
#         self.device = device

#         os.makedirs(self.save_dir, exist_ok=True)
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.writer = SummaryWriter(log_dir=self.log_dir)
    
#     def train(self):
#         for epoch in range(self.epochs):
#             self.model.train()
#             train_loss = 0.0
#             train_steps = 0

#             with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
#                 for idx, batch in enumerate(self.train_dataloader):
#                     loss, loss_dict = self.model.train_step(batch)
#                     train_loss += loss.item()
#                     train_steps += 1

#                     avg_train_loss_on_fly = train_loss / (idx + 1)

#                     pbar.set_postfix({
#                         'Loss': avg_train_loss_on_fly
#                     })
#                     pbar.update(1)

#                     if idx % 100 == 0:
#                         torch.cuda.empty_cache()
#                         gc.collect()
                    
#                     self.model.log_to_tensorboard(
#                         self.writer, loss_dict, train_steps + epoch * len(self.train_dataloader), prefix="train", batch=batch
#                     )
            
#             avg_train_loss = train_loss / train_steps
#             print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {avg_train_loss:.4f}")

#             if self.val_dataloader is not None:
#                 val_loss = self.model.validate(epoch, self.val_dataloader, self.writer)
#                 print(f"Epoch {epoch+1}/{self.epochs}, Average Val Loss: {val_loss:.4f}")
            
#             if (epoch + 1) % self.save_every == 0:
#                 self.model.save_checkpoint(epoch + 1)
        
#         self.writer.close()
