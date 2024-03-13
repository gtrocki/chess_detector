import torch
# from torchvision import models
from torch import nn

############################
# Piece classifier
############################

class PieceClassifier(nn.Module):
    def __init__(self):
        # super(PieceClassifier, self).__init__()
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=1),
        )
        self.linear = nn.Linear(64,6, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        Feature_maps = self.model(x)
        # Feature_maps_vector = Feature_maps.view(batch_size, -1)
        Feature_maps_vector = Feature_maps.reshape(batch_size, -1)
        y = self.linear(Feature_maps_vector)
        y = self.softmax(y)
        return y