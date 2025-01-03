from tqdm.notebook import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, Cycle

from .losses import feature_loss, discriminator_loss, generator_loss, melspec_loss

class Trainer:
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        model, 
        disc, 
        g_opt, 
        d_opt, 
        scheduler = None, 
        lambda_spec = 45,
        lambda_fm = 2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.disc = disc.to(device)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.scheduler = scheduler
        self.lambda_spec = lambda_spec
        self.lambda_fm = lambda_fm
        self.device = device
        self.writer = SummaryWriter(log_dir="../Tugas Akhir Kevin/logs/")
        
    def log_to_tensorboard(self, epoch, batch, g_loss, d_loss, mean_g_loss, mean_d_loss):
        self.writer.add_scalar('Mean Generator Loss', mean_g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Mean Discriminator loss', mean_d_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Generator Loss', g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Discriminator Loss', d_loss, epoch * len(self.train_loader) + batch)
    
    def start_training(self, epochs=10):
        self.model.train()
        self.disc.train()
        
        for epoch in range(epochs):
            mean_discriminator_loss, mean_generator_loss = 0, 0
            
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
                for i, (audio_arr, sr) in enumerate(self.train_loader):
                    audio_arr = audio_arr.to(self.device)
                    if len(audio_arr.shape) < 3:
                        audio_arr.unsqueeze(0)
                        
                    # Discriminator training
                    self.d_opt.zero_grad()
                    out, vq_results = self.model(audio_arr)
                    # print(audio_arr.shape)
                    audio_arr = audio_arr[:, :, :out.shape[-1]]
                    # print(f"audio_arr shape: {audio_arr.shape}, out shape: {out.shape}")
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.disc(audio_arr, out)
                    
                    disc_loss_value, _, _ = discriminator_loss(y_d_rs, y_d_gs)
                    
                    disc_loss_value.backward(retain_graph=True)
                    self.d_opt.step()

                    # Generator (Autoencoder) training
                    self.g_opt.zero_grad()
                    
                    _, y_d_gs, fmap_rs, fmap_gs = self.disc(audio_arr, out)
                    g_loss_value, _ = generator_loss(y_d_gs)
                    spec_loss = melspec_loss(audio_arr, out, self.model.spec_transform)
                    fm_loss = feature_loss(fmap_rs, fmap_gs)

                    gen_loss_value = g_loss_value + self.lambda_spec * spec_loss + self.lambda_fm * fm_loss
                    
                    gen_loss_value.backward()
                    self.g_opt.step()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()

                    mean_discriminator_loss += disc_loss_value.item()
                    mean_generator_loss += g_loss_value.item()

                    pbar.set_postfix({
                        'G_loss': mean_generator_loss / (i + 1),
                        'D_loss': mean_discriminator_loss / (i + 1)
                    })
                    pbar.update(1)

                    self.log_to_tensorboard(epoch, i, g_loss_value.item(), disc_loss_value.item(), mean_generator_loss / (i + 1), mean_discriminator_loss / (i + 1))
                    
            print(f"Epoch [{epoch+1}/{epochs}], G_loss: {mean_generator_loss:.4f}, D_loss: {mean_discriminator_loss:.4f}")

        print("Training complete")
