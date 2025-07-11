import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3LargeWithDropout(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV3LargeWithDropout, self).__init__()
        # ✅ Remove model wrapper — use model directly
        self.features = mobilenet_v3_large(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),  # MobileNet default
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
