from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.lldm_architecture import WaveLLDM
import gc

import torch
import os

class WaveLLDMTrainer:
    def __init__(
        self,
        model: WaveLLDM,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 300,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for idx, batch in enumerate(self.train_dataloader):
                    loss, loss_dict = self.model.train_step(batch)
                    train_loss += loss.item()
                    train_steps += 1

                    avg_train_loss_on_fly = train_loss / (idx + 1)

                    pbar.set_postfix({
                        'Loss': avg_train_loss_on_fly
                    })
                    pbar.update(1)

                    if idx % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    self.model.log_to_tensorboard(
                        self.writer, loss_dict, train_steps + epoch * len(self.train_dataloader), prefix="train", batch=batch
                    )
            
            avg_train_loss = train_loss / train_steps
            print(f"Epoch {epoch+1}/{self.epochs}, Average Train Loss: {avg_train_loss:.4f}")

            if self.val_dataloader is not None:
                val_loss = self.model.validate(epoch, self.val_dataloader, self.writer)
                print(f"Epoch {epoch+1}/{self.epochs}, Average Val Loss: {val_loss:.4f}")
            
            if (epoch + 1) % self.save_every == 0:
                self.model.save_checkpoint(epoch + 1)
        
        self.writer.close()
