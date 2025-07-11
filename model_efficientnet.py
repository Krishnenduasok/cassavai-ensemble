
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

class EfficientNetV2SWithDropout(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetV2SWithDropout, self).__init__()
        self.model = efficientnet_v2_s(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)
