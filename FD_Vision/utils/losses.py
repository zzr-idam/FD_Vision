import torch
import torch.nn as nn
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.layer1 = nn.Sequential(*vgg[:2])
        self.layer2 = nn.Sequential(*vgg[2:7])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        l1 = torch.abs(self.layer1(x) - self.layer1(y)).mean()
        l2 = torch.abs(self.layer2(x) - self.layer2(y)).mean()
        return l1 + l2


class FDVMLoss(nn.Module):
    def __init__(self, lambda_per=0.5):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_per = lambda_per

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        return l1 + self.lambda_per * perceptual