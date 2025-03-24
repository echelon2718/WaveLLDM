from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import os
import gc
from .losses import feature_loss, discriminator_loss, generator_loss, melspec_loss, MultiScaleSpecLoss, MultiScaleSTFTLoss, EnhancedMultiScaleSTFTLoss

class Trainer:
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        model, 
        disc, 
        g_opt, 
        d_opt, 
        scheduler=None, 
        lambda_spec=30,
        lambda_stft=20,
        lambda_fm=2,
        save_interval=10000,
        train_stage=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.disc = disc.to(device)
        self.multiscale_melspec_loss = MultiScaleSpecLoss()
        self.multiscale_stft_loss = EnhancedMultiScaleSTFTLoss()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.scheduler = scheduler
        self.lambda_spec = lambda_spec
        self.lambda_stft = lambda_stft
        self.lambda_fm = lambda_fm
        self.save_interval = save_interval
        self.train_stage = train_stage
        self.device = device
        self.writer = SummaryWriter(log_dir="../Tugas Akhir Kevin/logs/")
        self.global_step = 0

    def save_model(self):
        torch.save(self.model.state_dict(), f"../Tugas Akhir Kevin/results/generator_step_{self.global_step}.pth")
        torch.save(self.disc.state_dict(), f"../Tugas Akhir Kevin/results/discriminator_step_{self.global_step}.pth")

    def validate(self):
        self.model.eval()
        self.disc.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_item in self.val_loader:
                # Pilih tensor audio untuk validasi; pada stage 2, bisa disesuaikan jika ingin menggunakan noisy sebagai input validasi
                if self.train_stage == 1:
                    audio_arr = val_item["clean_audio_tensor"]
                elif self.train_stage == 2:
                    audio_arr = val_item["clean_audio_tensor"]
                    
                audio_arr = audio_arr.to(self.device, non_blocking=True)
                if len(audio_arr.shape) < 3:
                    audio_arr = audio_arr.unsqueeze(0)

                out, _ = self.model(audio_arr)
                audio_arr = audio_arr[:, :, :out.shape[-1]]  # Ensure same shape as output
                
                # Menggunakan discriminator untuk menghitung loss generator pada validasi
                _, y_d_gs, fmap_rs, fmap_gs = self.disc(audio_arr, out)
                g_loss_value, _ = generator_loss(y_d_gs)
                spec_loss = self.multiscale_melspec_loss(audio_arr, out, self.device)
                stft_loss = self.multiscale_stft_loss(audio_arr, out, self.device)
                fm_loss = feature_loss(fmap_rs, fmap_gs)

                # Total loss validasi merupakan kombinasi loss generator dan loss tambahan
                val_loss = g_loss_value + self.lambda_spec * spec_loss + self.lambda_stft * stft_loss + self.lambda_fm * fm_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.writer.add_scalar('Validation Loss', avg_val_loss, self.global_step)
        # Kembalikan mode training
        self.model.train()
        self.disc.train()
        return avg_val_loss
    
    def start_training(self, epochs=10):
        self.model.train()
        self.disc.train()

        for epoch in range(epochs):
            mean_discriminator_loss, mean_generator_loss = 0, 0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
                for i, item in enumerate(self.train_loader):
                    if self.train_stage == 1:
                        audio_arr = item["clean_audio_tensor"]
                    elif self.train_stage == 2:
                        audio_arr = item["clean_audio_tensor"] if i % 2 == 0 else item["noisy_audio_tensor"]
                        
                    self.global_step += 1

                    audio_arr = audio_arr.to(self.device, non_blocking=True)
                    if len(audio_arr.shape) < 3:
                        audio_arr = audio_arr.unsqueeze(0)

                    # Discriminator training
                    self.d_opt.zero_grad()
                    with torch.no_grad():  # No gradient required for discriminator inputs
                        out, _ = self.model(audio_arr)
                        audio_arr = audio_arr[:, :, :out.shape[-1]]  # Ensure same shape as output

                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.disc(audio_arr, out)
                    disc_loss_value, _, _ = discriminator_loss(y_d_rs, y_d_gs)
                    disc_loss_value.backward()
                    self.d_opt.step()

                    # Generator (Autoencoder) training
                    self.g_opt.zero_grad()
                    out, _ = self.model(audio_arr)  # Recompute generator output for gradients
                    _, y_d_gs, fmap_rs, fmap_gs = self.disc(audio_arr, out)

                    g_loss_value, _ = generator_loss(y_d_gs)
                    spec_loss = self.multiscale_melspec_loss(audio_arr, out, self.device)
                    stft_loss = self.multiscale_stft_loss(audio_arr, out, self.device)
                    fm_loss = feature_loss(fmap_rs, fmap_gs)

                    gen_loss_value = g_loss_value + self.lambda_spec * spec_loss + self.lambda_stft * stft_loss + self.lambda_fm * fm_loss
                    gen_loss_value.backward()
                    self.g_opt.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    mean_discriminator_loss += disc_loss_value.item()
                    mean_generator_loss += g_loss_value.item()

                    avg_disc_loss = mean_discriminator_loss / (i + 1)
                    avg_gen_loss = mean_generator_loss / (i + 1)

                    pbar.set_postfix({
                        'G_loss': avg_gen_loss,
                        'D_loss': avg_disc_loss
                    })
                    pbar.update(1)

                    # Logging nilai loss rata-rata ke TensorBoard
                    self.writer.add_scalar('Generator Loss', g_loss_value.item(), self.global_step)
                    self.writer.add_scalar('Mel Spectogram Loss', spec_loss.item(), self.global_step)
                    self.writer.add_scalar('STFT Loss', stft_loss.item(), self.global_step)
                    self.writer.add_scalar('Feature Matching Loss', fm_loss.item(), self.global_step)
                    self.writer.add_scalar('Discriminator Loss', disc_loss_value.item(), self.global_step)
                    self.writer.add_scalar('Mean Generator Loss', avg_gen_loss, self.global_step)
                    self.writer.add_scalar('Mean Discriminator Loss', avg_disc_loss, self.global_step)
                    
                    if i % 100 == 0:
                        self.writer.add_audio('Generated Audio', out[0].detach().cpu()[0].numpy(), self.global_step, sample_rate=44100)
                    
                    if i % self.save_interval == 0:
                        self.save_model()

                    # Periodic memory cleanup
                    if i % 500 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            if self.val_loader is None:
                print(f"Epoch [{epoch+1}/{epochs}], G_loss: {avg_gen_loss:.4f}, D_loss: {avg_disc_loss:.4f}")
            else:
                # Setelah setiap epoch, hitung loss validasi
                avg_val_loss = self.validate()
                print(f"Epoch [{epoch+1}/{epochs}], G_loss: {avg_gen_loss:.4f}, D_loss: {avg_disc_loss:.4f}, Val_loss: {avg_val_loss:.4f}")

        print("Training complete")
