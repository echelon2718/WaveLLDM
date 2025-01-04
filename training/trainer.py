from tqdm.notebook import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

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
        scheduler=None, 
        lambda_spec=45,
        lambda_fm=2,
        save_interval=10000,
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
        self.save_interval = save_interval
        self.device = device
        self.writer = SummaryWriter(log_dir="../Tugas Akhir Kevin/logs/")
        self.global_step = 0

    def log_to_tensorboard(self, epoch, batch, g_loss, d_loss, mean_g_loss, mean_d_loss, audio_arr, out):
        self.writer.add_scalar('Mean Generator Loss', mean_g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Mean Discriminator Loss', mean_d_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Generator Loss', g_loss, epoch * len(self.train_loader) + batch)
        self.writer.add_scalar('Discriminator Loss', d_loss, epoch * len(self.train_loader) + batch)
    
        # Prepare audio samples
        if batch == 0:  # Log only once per epoch
            input_audio = audio_arr[0].squeeze().cpu()  # Remove batch and channel dimensions
            generated_audio = out[0].squeeze().detach().cpu()
            
            if input_audio.dim() == 1:  # Add channel dimension if missing
                input_audio = input_audio.unsqueeze(1)
            if generated_audio.dim() == 1:
                generated_audio = generated_audio.unsqueeze(1)
    
            self.writer.add_audio('Input Audio', input_audio, epoch, sample_rate=44100)
            self.writer.add_audio('Generated Audio', generated_audio, epoch, sample_rate=44100)


    def save_model(self):
        torch.save(self.model.state_dict(), f"../Tugas Akhir Kevin/results/generator_step_{self.global_step}.pth")
        torch.save(self.disc.state_dict(), f"../Tugas Akhir Kevin/results/discriminator_step_{self.global_step}.pth")

    def start_training(self, epochs=10):
        self.model.train()
        self.disc.train()

        for epoch in range(epochs):
            mean_discriminator_loss, mean_generator_loss = 0, 0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
                for i, (audio_arr, sr) in enumerate(self.train_loader):
                    self.global_step += 1

                    audio_arr = audio_arr.to(self.device)
                    if len(audio_arr.shape) < 3:
                        audio_arr.unsqueeze(0)

                    # Discriminator training
                    self.d_opt.zero_grad()
                    out, vq_results = self.model(audio_arr)
                    audio_arr = audio_arr[:, :, :out.shape[-1]]
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

                    self.log_to_tensorboard(epoch, i, g_loss_value.item(), disc_loss_value.item(), mean_generator_loss / (i + 1), mean_discriminator_loss / (i + 1), audio_arr, out)

                    # Save model every save_interval steps
                    if self.global_step % self.save_interval == 0:
                        self.save_model()

            print(f"Epoch [{epoch+1}/{epochs}], G_loss: {mean_generator_loss:.4f}, D_loss: {mean_discriminator_loss:.4f}")

        print("Training complete")
