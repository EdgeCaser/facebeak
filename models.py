import torch
import torch.nn as nn
import torchvision

class CrowResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # Remove classifier
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize for triplet loss
        return x 