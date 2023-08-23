import torch
import torch.nn as nn
import math
import cv2
from torch.autograd import Variable


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]), 2).sum()

        return self.TVLoss_weight*(h_tv / count_h + w_tv / count_w) / batch_size


class TVLossGradient(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLossGradient,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, x, y):
        batch_size = x.size()[0]

        count_h_x = self._tensor_size(x[:,:,1:,:])
        count_w_x = self._tensor_size(x[:,:,:,1:])

        h_tv_x = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :])) # |G_h(x)|
        w_tv_x = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1])) # |G_v(x)|

        h_tv_x = h_tv_x ** 2  #||G_h(x)||^2
        w_tv_x = w_tv_x ** 2  #||G_v(x)||^2

        h_tv_y = torch.abs((y[:, :, 1:, :] - y[:, :, :-1, :])) # |G_h(y)|
        w_tv_y = torch.abs((y[:, :, :, 1:] - y[:, :, :, :-1])) # |G_v(y)|

        h_tv_y = torch.pow(h_tv_y, 1.2) + 0.0001  #||G_h(y)||^1.2
        w_tv_y = torch.pow(w_tv_y, 1.2) + 0.0001  #||G_v(y)||^1.2

        ### ((||G_h(x)||^2 / ||G_h(y)||^1.2) + (||G_v(x)||^2 / ||G_v(y)||^1.2)) / (H*W)
        TVloss = torch.abs(h_tv_x / h_tv_y).sum() / count_h_x + torch.abs(w_tv_x / w_tv_y).sum() / count_w_x

        return self.TVLoss_weight * TVloss / batch_size


