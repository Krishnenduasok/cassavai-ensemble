import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
class_names = ['high', 'low', 'md', 'medium', 'zero']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# ResNet50 with dropout
class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# EfficientNetV2S with dropout
class EfficientNetV2SWithDropout(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Ensemble prediction function
def predict_ensemble(image_path, resnet, efficientnet, mobilenet, yolo_model, return_probs=False, verbose=True):
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        resnet_probs = F.softmax(resnet(input_tensor.to("cpu")), dim=1)
        efficientnet_probs = F.softmax(efficientnet(input_tensor.to("cpu")), dim=1)
        mobilenet_probs = F.softmax(mobilenet(input_tensor.to("cpu")), dim=1)

        yolo_results = yolo_model(image_path, verbose=False)
        yolo_raw_probs = yolo_results[0].probs.data.tolist()
        yolo_probs = torch.tensor(yolo_raw_probs).unsqueeze(0).to("cpu")

        avg_probs = (
            0.2 * resnet_probs +
            0.2 * efficientnet_probs +
            0.2 * mobilenet_probs +
            0.4 * yolo_probs
        )

        final_pred = torch.argmax(avg_probs, dim=1).item()

        # Always compute individual top classes (needed even if verbose=False)
        r_top = torch.argmax(resnet_probs).item()
        e_top = torch.argmax(efficientnet_probs).item()
        m_top = torch.argmax(mobilenet_probs).item()
        y_top = torch.argmax(yolo_probs).item()

        logs = []
        logs.append("📊 Individual Model Predictions:")
        logs.append(f"  • ResNet50       → {class_names[r_top]} ({resnet_probs[0][r_top]:.2f})")
        logs.append(f"  • EfficientNetV2 → {class_names[e_top]} ({efficientnet_probs[0][e_top]:.2f})")
        logs.append(f"  • MobileNetV3    → {class_names[m_top]} ({mobilenet_probs[0][m_top]:.2f})")
        logs.append(f"  • YOLOv8         → {class_names[y_top]} ({yolo_probs[0][y_top]:.2f})")
        logs.append("\n📈 Averaged Probabilities:")
        for i, cls in enumerate(class_names):
            logs.append(f"  {cls:<8} : {avg_probs[0][i]:.4f}")
        logs.append(f"\n✅ Final PPD Prediction (Soft Voting): {class_names[final_pred]}")

    if return_probs:
        return final_pred, avg_probs.squeeze().tolist(), "\n".join(logs)

    return final_pred, None, "\n".join(logs)
