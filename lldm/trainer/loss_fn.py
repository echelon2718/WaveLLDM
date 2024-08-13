import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdversarialLoss(nn.Module):
    def __init__(self, device="cuda"):
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
    def __init__(self, disc,device="cuda"):
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
                 lambda_adv=0.1, 
                 lambda_recon=0.1, 
                 lambda_fm=0.1, 
                 lambda_kl=0.1, 
        ):
        super(GeneratorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss()
        self.recon_loss = nn.L1Loss()
        self.feature_matching = FeatureMatchingLoss(disc=fm_disc)

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
    def __init__(self, device="cuda"):
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