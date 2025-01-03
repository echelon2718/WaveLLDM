import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

def feature_loss(fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def melspec_loss(aud_x, aud_y, spec_transform):
    # Ensure same length in temporal dimension
    aud_x, aud_y = aud_x.to(device), aud_y.to(device)
    if aud_x.shape[-1] > aud_y.shape[-1]:
        aud_y = F.pad(aud_y, (0, aud_x.shape[-1] - aud_y.shape[-1]))
    elif aud_x.shape[-1] < aud_y.shape[-1]:
        aud_x = F.pad(aud_x, (0, aud_y.shape[-1] - aud_x.shape[-1]))
    
    # Compute mel-spectrograms
    spec_x = spec_transform(aud_x)
    spec_y = spec_transform(aud_y)
    
    # Calculate L1 loss
    loss = nn.L1Loss()(spec_x, spec_y)
    return loss

class AdversarialLoss(nn.Module):
    def __init__(self, device=device):
        super(AdversarialLoss, self).__init__()
        self.device = device
        self.loss = nn.BCELoss()

    def forward(self, X, gen, disc):
        X = X.to(self.device)
        recon_X, posterior = gen(X)
        preds = disc(recon_X)
        loss = self.loss(preds, torch.ones_like(preds))
        return loss, recon_X, posterior

class FeatureMatchingLoss(nn.Module):
    def __init__(self, disc,device=device):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.disc = disc.to(device)

    def forward(self, real_data, fake_data):
        loss = 0
        with torch.no_grad():
            _, real_features = self.disc(real_data, return_features=True)
            _, fake_features = self.disc(fake_data, return_features=True)
        
        for real, fake in zip(real_features, fake_features):
            loss += self.criterion(fake, real.detach())
            
        return loss

class GeneratorLoss(nn.Module):
    def __init__(self,
                 fm_disc,
                 lambda_adv=1, 
                 lambda_fm=2, 
                 lambda_spec=45, 
        ):
        super(GeneratorLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_recon = lambda_recon
        self.lambda_fm = lambda_fm
        self.lambda_kl = lambda_kl

    def forward(self, X, gen, disc):
        adv_loss, recon_X, posterior = self.adversarial_loss(X, gen, disc)
        recon_loss = self.recon_loss(X, recon_X)
        fm_loss = self.feature_matching(X, recon_X)
        kl_loss = posterior.KL()

        loss = self.lambda_adv * adv_loss + self.lambda_recon * recon_loss + self.lambda_fm * fm_loss + self.lambda_kl * kl_loss
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, device=device):
        super(DiscriminatorLoss, self).__init__()
        self.device = device
        self.loss = nn.BCELoss()

    def forward(self, real_Y, fake_Y, disc):
        real_Y, fake_Y = real_Y.to(self.device), fake_Y.to(self.device)
        preds_real = disc(real_Y)
        loss_real = self.loss(preds_real, torch.ones_like(preds_real))  # Target is 1 for real

        preds_fake = disc(fake_Y)
        loss_fake = self.loss(preds_fake, torch.zeros_like(preds_fake))  # Target is 0 for fake

        return loss_real + loss_fake
