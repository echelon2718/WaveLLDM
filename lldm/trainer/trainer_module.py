from tqdm.notebook import tqdm
import torch
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, train_loader, test_loader, model, disc, g_loss, d_loss, g_opt, d_opt, scheduler, device="cuda"):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.disc = disc.to(device)
        self.scheduler = scheduler
        self.g_loss = g_loss.to(device)
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.device = device
        self.writer = SummaryWriter(log_dir="/drive0-storage/Kevin_Putra_Santoso/AudioLLDM/logs")
        
    def log_to_tensorboard(self, epoch, batch, g_loss, d_loss, mean_g_loss, mean_d_loss):
        self.writer.add_scalar('Mean Generator Loss', mean_g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Mean Discriminator loss', mean_d_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Generator Loss', g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Discriminator Loss', d_loss, epoch * len(self.train_loader) + batch)
    
    def start_training(self, epochs=10):
        for epoch in range(epochs):
            mean_discriminator_loss, mean_generator_loss = 0, 0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
                for i, spec in enumerate(self.train_loader):
                    spec = spec.to(self.device)

                    # Discriminator training
                    self.d_opt.zero_grad()
                    out, posterior = self.model(spec)
                    disc_loss_value = self.d_loss(spec, out, self.disc)
                    disc_loss_value.backward()
                    self.d_opt.step()

                    # Generator (Autoencoder) training
                    self.g_opt.zero_grad()
                    g_loss_value = self.g_loss(spec, self.model, self.disc)[0]
                    g_loss_value.backward()
                    self.scheduler.step()
                    self.g_opt.step()

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