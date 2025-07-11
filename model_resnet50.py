import torch.nn as nn
from torchvision import models

class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50WithDropout, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)
