import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.ssimloss import SSIMLoss

class SSIMAttentionLoss(nn.Module):
    def __init__(self):
        super(SSIMAttentionLoss, self).__init__()
        self.alpha  = 1.0
        self.beta   = 0.0
        self.maxVal = 3.0

        self.ssimLoss = SSIMLoss().cuda()
        self.L1loss = nn.L1Loss().cuda()

    def forward(self, step, tgt, src):

        sigma = self.alpha * step + self.beta
        gauss = lambda x: torch.exp(-((x + 1) / sigma) ** 2) * self.maxVal
        ssim_map = self.ssimLoss(tgt, src, reduction='none').detach()
        gauss_map = gauss(ssim_map).detach()
        new_src, new_tgt = src * gauss_map, tgt * gauss_map

        loss = self.L1loss(new_src, new_tgt)

        return loss
