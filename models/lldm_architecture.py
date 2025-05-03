import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from inspect import isfunction
import os
import math

### UTIL FUNCTIONS ###  
def extract_into_tensor(a, t, x_shape):
    '''
    Extracts a tensor from a larger tensor based on the indices provided in t.
    The output tensor has the same shape as the input tensor, but with the dimensions of t replaced by 1s.
    The output tensor is reshaped to match the input tensor's shape.

    Parameters:
        a (torch.Tensor): The larger tensor from which to extract values.
        t (torch.Tensor): The indices used to extract values from a.
        x_shape (tuple): The shape of the input tensor.
    '''
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    '''
    Generates noise of the same shape as the input tensor.
    If repeat is True, it generates noise of the same shape as the first dimension of the input tensor.
    Otherwise, it generates noise of the same shape as the input tensor.

    Parameters:
        shape (tuple): Shape of the input tensor.
        device (torch.device): Device to generate the noise on.
        repeat (bool): If True, generates noise of the same shape as the first dimension of the input tensor.
    '''
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def exists(x):
    '''
    Checks if the input x is not None.
    '''
    return x is not None

def default(val, d):
    '''
    Returns the value of val if it exists, otherwise returns the default value d.
    If d is a function, it calls the function and returns its value.
    
    '''
    if exists(val):
        return val
    return d() if isfunction(d) else d

######################################

############ EMA Helper ############

class EMA:
    '''
    Exponential Moving Average (EMA) class for model parameters.
    This class is used to maintain a moving average of model parameters during training.
    It helps in stabilizing the training process and improving the performance of the model.

    Parameters:
        model (nn.Module): The model whose parameters are to be averaged.
        decay (float): The decay rate for the moving average. Default is 0.9999.
    '''
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
    ):
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
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])
    
    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.shadow[name] * self.decay + param.data * (1 - self.decay)

class DDPM(nn.Module):
    def __init__(
        self,
        p_estimator: nn.Module, # model unet sebagai estimator
        num_timesteps: int = 1000, # jumlah timestep maksimum denoising
        beta_scheduler: str = "linear", # scheduler beta untuk mengatur intensitas noise, opsi: linear, cosine, sqrt_linear, sqrt
        loss_type: str = "l2", # loss function yang digunakan, l2 = MSE, l1 = MAE
        log_every: int = 100, # frekuensi logging
        use_ema: bool = True, # menggunakan exponential moving average untuk mengatur parameter model selama training
        clip_denoised=True, # apakah akan meng-clamp hasil denoising ke rentang [-1, 1]
        ema_decay: float = 0.9999, # decay rate untuk EMA
        linear_start: float = 1e-4, # nilai beta awal untuk scheduler linear
        linear_end: float = 2e-2, # nilai beta akhir untuk scheduler linear
        cosine_s: float = 0.008, # parameter untuk cosine scheduler (konteks beta scheduler, bukan learning rate)
        given_betas: list = None, # jika diberikan, gunakan beta yang sudah ada
        l_simple_weight: float = 1., # bobot untuk loss sederhana
        original_elbo_weight: float = 0.0001, # bobot untuk loss ELBO asli
        conditioning_key: str = "z_noisy", # kunci untuk data yang digunakan untuk conditioning (misalnya, "z_noisy" untuk denoising)
        parameterization: str = "eps", # parametrisasi untuk noise (eps atau x0)
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
        recon_loss_weight: float = 0.0,
        device: int = int(os.environ["LOCAL_RANK"]), # perangkat untuk model (CPU atau GPU)
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.loss_type = loss_type
        self.device = device
        self.recon_loss_weight = recon_loss_weight
        self.p_estimator = p_estimator
        self.use_ema = use_ema

        # ================= parameter untuk difusi ================= #
        self.num_timesteps = num_timesteps
        self.beta_scheduler = beta_scheduler
        self.betas = given_betas if given_betas is not None else self.get_beta_schedule(
            beta_schedule="cosine",
            num_timesteps=num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        
        self.alphas                         = 1.0 - self.betas
        self.alphas_cumprod                 = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev            = np.append(1., self.alphas_cumprod[:-1])

        # kalkulasi untuk menghitung q(x_t | x_{t-1}) dan lain-lain
        self.sqrt_alphas_cumprod            = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod  = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod   = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod      = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod    = np.sqrt(1. / self.alphas_cumprod - 1)

        # kalkulasi untuk posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance             = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1           = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2           = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )


        # KONVERSI KE TENSOR DAN PENGATURAN PERANGKAT (GPU/CPU)
        self.betas                          = torch.tensor(self.betas, dtype=torch.float32).to(device)
        self.alphas                         = torch.tensor(self.alphas, dtype=torch.float32).to(device)
        self.alphas_cumprod                 = torch.tensor(self.alphas_cumprod, dtype=torch.float32).to(device)
        self.alphas_cumprod_prev            = torch.tensor(self.alphas_cumprod_prev, dtype=torch.float32).to(device)
        self.sqrt_alphas_cumprod            = torch.tensor(self.sqrt_alphas_cumprod, dtype=torch.float32).to(device)
        self.sqrt_one_minus_alphas_cumprod  = torch.tensor(self.sqrt_one_minus_alphas_cumprod, dtype=torch.float32).to(device)
        self.log_one_minus_alphas_cumprod   = torch.tensor(self.log_one_minus_alphas_cumprod, dtype=torch.float32).to(device)
        self.sqrt_recip_alphas_cumprod      = torch.tensor(self.sqrt_recip_alphas_cumprod, dtype=torch.float32).to(device)
        self.sqrt_recipm1_alphas_cumprod    = torch.tensor(self.sqrt_recipm1_alphas_cumprod, dtype=torch.float32).to(device)
        self.posterior_variance             = torch.tensor(self.posterior_variance, dtype=torch.float32).to(device)
        self.posterior_log_variance_clipped = torch.tensor(self.posterior_log_variance_clipped, dtype=torch.float32).to(device)
        self.posterior_mean_coef1           = torch.tensor(self.posterior_mean_coef1, dtype=torch.float32).to(device)
        self.posterior_mean_coef2           = torch.tensor(self.posterior_mean_coef2, dtype=torch.float32).to(device)

        self.original_elbo_weight = original_elbo_weight
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * self.alphas * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(self.alphas_cumprod) / (2. * 1 - self.alphas_cumprod)
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()

        # ======================================================== #

        self.log_every = log_every
        self.ema_decay = ema_decay
        
        self.l_simple_weight = l_simple_weight
        self.conditioning_key = conditioning_key
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
    
    def get_beta_schedule(
        self, 
        beta_schedule : str, 
        num_timesteps : int, 
        linear_start : float, 
        linear_end : float, 
        cosine_s : float
    ):
        if beta_schedule == "linear":
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float64) ** 2
            return betas.numpy()

        elif beta_schedule == "cosine":
            betas = self.beta_for_alpha_bar(
                num_timesteps,
                lambda t: math.cos((t + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2,
            )

            return betas
        
        elif beta_schedule == "sqrt_linear":
            betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float64)
            return betas.numpy()
        
        elif beta_schedule == "sqrt":
            betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float64) ** 0.5
            return betas.numpy()
        
        else:
            raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")

    def beta_for_alpha_bar(
        self, 
        num_timesteps : int, 
        alpha_bar, 
        max_beta : float = 0.999
    ):
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        
        return np.array(betas)
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        Parameters:
            x_start: the [N x C x ...] tensor of noiseless inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        Parameters:
            x_start: the initial data batch.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            noise: if specified, the split-out normal noise.

        Returns: A noisy version of x_start.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) 
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Get the posterior distribution q(x_{t-1} | x_t, x_0).
        This is the distribution we will sample from to get x_{t-1} from x_t.

        Parameters:
            x_start: the [N x C x ...] tensor of noiseless inputs.
            x_t: the [N x C x ...] tensor of noisy inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, y, clip_denoised: bool):
        """
        Get the distribution p(x_{t-1} | x_t).
        This is the distribution we will sample from to get x_{t-1} from x_t.

        Parameters:
            x: the [N x C x ...] tensor of noisy inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            clip_denoised: whether to clip the denoised output to [-1, 1].
            return: A tuple (mean, variance, log_variance), all of x's shape.
        """
        model_out = self.p_estimator(x, t, y)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        elif self.parameterization == "x0":
            x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def predict_start_from_noise(self, x_t, t, noise):
        ''' 
        Implement the formula: x₀ = (1 / sqrt(αₜ)) * (xₜ - sqrt(1 - ᾱₜ) * ϵ)
        '''
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)
    
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()

        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def p_losses(self, x_start, t, noise=None, add_recon_loss=False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        print("Noise shape: ", noise.shape)
        print("x_start shape: ", x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.p_estimator(x_noisy, t)
        print("Model out shape: ", model_out.shape)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        if add_recon_loss:
            # Compute reconstruction loss
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=model_out)
            if self.learn_logvar:
                logvar = self.logvar[t]
                recon_loss = 0.5 * (logvar + ((x_start - x_recon) ** 2) / torch.exp(logvar)).mean()
            else:
                recon_loss = self.get_loss(x_start, x_recon, mean=True)
            
            loss_dict.update({f'{log_prefix}/loss_recon': recon_loss})
            loss_simple += self.recon_loss_weight * recon_loss
        
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
    
    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)
    
class WaveLLDM(DDPM):
    def __init__(
            self,
            encoder: nn.Module, # model encoder untuk mengubah data ke latent space
            decoder: nn.Module, # model decoder untuk mengubah data dari latent space ke data asli
            quantizer, # model quantizer untuk mengubah data ke representasi diskrit
            std_scale_factor: float = 1.0, # apakah akan melakukan rescaling pada latent space. Default 0.5061
            z_dim: int = 512, # dimensi latent space
            ema_decay: float = 0.9999, # decay rate untuk EMA
            use_latent: bool = True, # apakah akan menggunakan latent space untuk training
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.log_every_t = kwargs.get('log_every_t', 100) # log every t steps
        self.beta_scheduler = kwargs.get('beta_scheduler', 'cosine') # beta scheduler untuk mengatur intensitas noise
        self.original_elbo_weight = kwargs.get('original_elbo_weight', 1.0) # bobot untuk original elbo loss

        self.std_scale_factor = std_scale_factor
        self.z_dim = z_dim
        self.use_latent = use_latent

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.ema = EMA(self.p_estimator, decay=ema_decay)
        self.quantizer = quantizer.to(self.device)

        # Freeze pretrained components
        self.encoder.requires_grad_(False)
        self.quantizer.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.encoder.eval()
        self.quantizer.eval()
        self.decoder.eval()

    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input data to latent space.
        :param x: input data
        :return: encoded data in latent space
        '''
        assert x.dim() == 3, "Input data must be 3D tensor (batch_size, channels, length)"
        z = self.encoder(x)

        if self.use_latent:
            z = self.quantizer(z).latents
        else:
            z = self.quantizer(z).z
        
        return z
    
    @torch.no_grad()
    def decode(self, z):
        '''
        Decode the latent space data to original data.
        :param z: latent space data
        :return: decoded data
        '''
        assert z.dim() == 3, "Latent data must be 3D tensor (batch_size, channels, length)"
        x_recon = self.decoder(z)
        
        return x_recon
    
    def p_mean_variance(self, x, t, y, clip_denoised: bool):
        model_out = self.p_estimator(x, t, y)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def p_losses(self, z_start, t, y, noise=None, add_recon_loss=False):
        # Generate noise if not provided
        noise = default(noise, lambda: torch.randn_like(z_start))
        # Sample noisy latents from q(x_t | x_0)
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        # Predict noise (or x0) conditioned on degraded latents y
        model_out = self.p_estimator(z_noisy, t, y)
        
        loss_dict = {}
        # Determine target based on parameterization
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = z_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not supported")
        
        # Compute loss
        loss = self.get_loss(model_out, target, mean=True)
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss})
        loss = loss * self.l_simple_weight
        return loss, loss_dict
    
    @torch.no_grad()
    def sample_with_ema(self, degraded_audio, batch_size=1):
        # Encode degraded audio to latent space
        if degraded_audio.shape[1] != 512:
            y = self.encode(degraded_audio)
        else:
            y = degraded_audio
        
        shape = (batch_size, self.z_dim, *y.shape[2:])  # Match latent dimensions
        
        # Apply EMA parameters to p_estimator
        self.ema.apply(self.p_estimator)
        z_sample = self.p_sample_loop(shape, y)  # Note: p_sample_loop needs y
        # Restore original parameters
        for name, param in self.p_estimator.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema.original[name])
        
        return z_sample

    @torch.no_grad()
    def p_sample(self, x, t, y, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, y, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, y, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), y,
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def forward(self, batch):
        # Encode clean and degraded audio to latent space
        clean_latents = batch["clean_audio_downsampled_latents"]
        degraded_latents = batch["noisy_audio_downsampled_latents"]

        if self.std_scale_factor: # Rescale latents if std_scale_factor is provided (1/std_scale_factor * latents)
            clean_latents = clean_latents * (1./self.std_scale_factor)
            degraded_latents = degraded_latents * (1./self.std_scale_factor)
        
        # Sample timestep t
        t = torch.randint(0, self.num_timesteps, (clean_latents.shape[0],), device=self.device).long()
        
        # Compute loss using p_losses, conditioned on degraded_latents
        loss, loss_dict = self.p_losses(clean_latents, t, degraded_latents, add_recon_loss=False)
        return loss, loss_dict
