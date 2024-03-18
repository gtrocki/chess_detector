import torch
from torchvision import models
from torch import nn

############################
# Pawn classifier
############################

class ResnetPawnClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(num_classes=num_classes)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x