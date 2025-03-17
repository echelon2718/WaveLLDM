import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm

class BetaScheduler:
    """Linear beta scheduler untuk proses diffusion"""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def sample_timesteps(self, batch_size):
        return torch.randint(1, self.T, (batch_size,))

class EMA:
    """Exponential Moving Average untuk model parameters"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.shadow[name] * self.decay + param.data * (1 - self.decay)

class DiffusionTrainer:
    def __init__(self, config):
        self.device = config['device']
        self.model = create_diffusion_model(
            in_channels=config['in_channels'],
            base_channels=config['base_channels'],
            out_channels=config['out_channels']
        ).to(self.device)
        
        self.ema = EMA(self.model, decay=config['ema_decay'])
        self.ema_model = create_diffusion_model(**config).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.scheduler = BetaScheduler(T=config['T'])
        self.T = config['T']
        self.loss_fn = nn.MSELoss()
        
    def forward_process(self, x0, t):
        """Tambahkan noise ke data input sesuai timestep t"""
        sqrt_alpha_bar = torch.sqrt(self.scheduler.alpha_bars[t][:, None, None])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.scheduler.alpha_bars[t][:, None, None])
        
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        x0 = batch.to(self.device)  # Input shape (B, d_emb, seq_len)
        B = x0.size(0)
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(B).to(self.device)
        
        # Forward process
        xt, noise = self.forward_process(x0, t)
        
        # Prediksi noise
        pred_noise = self.model(xt, t)
        
        # Hitung loss
        loss = self.loss_fn(pred_noise, noise)
        loss.backward()
        
        # Update model
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update EMA
        self.ema.update(self.model)
        
        return loss.item()
    
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                total_loss += loss
                progress_bar.set_postfix(loss=loss)
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint dengan EMA model
            self.ema.apply(self.ema_model)
            torch.save({
                'model': self.model.state_dict(),
                'ema_model': self.ema_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, f"diffusion_checkpoint_epoch{epoch+1}.pth")
            self.ema.apply(self.model)  # Kembalikan parameter original
