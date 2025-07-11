import streamlit as st
from PIL import Image
import gdown
import os
import torch
from torchvision import transforms
from model_resnet50 import ResNet50WithDropout
from model_efficientnet import EfficientNetV2SWithDropout
from model_mobilenetv3 import MobileNetV3LargeWithDropout
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
from ensemble_predict import predict_ensemble, class_names
import sys
import streamlit as st
st.write("Python Version:", sys.version)

# MobileNetV3 structure
def get_mobilenet_v3(num_classes=5):
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# File download helper
def download_if_needed(file_id, output_path):
    if not os.path.exists(output_path):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False, fuzzy=True)
        except Exception as e:
            raise RuntimeError(f"Download failed for {output_path}: {e}")

# Load models
@st.cache_resource
def load_models():
    models_dict = {}

    try:
        models_dict['YOLOv8'] = YOLO("cassava_ppd_yolov8.pt")
    except Exception as e:
        st.warning(f"⚠️ YOLOv8 load failed: {e}")

    try:
        resnet_path = "resnetfinal_state_dict.pth"
        download_if_needed("11Kwodly2XNUcOt7HdlBbD77sTsC4_I9o", resnet_path)
        resnet_model = ResNet50WithDropout(num_classes=len(class_names))
        resnet_model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
        resnet_model.eval()
        models_dict['ResNet50'] = resnet_model
    except Exception as e:
        st.warning(f"⚠️ ResNet50 load failed: {e}")

    try:
        eff_path = "efficientnet_state_dict.pth"
        download_if_needed("10wlsWr-St47LCUQ7wGqH5BecrGPJHJwL", eff_path)
        eff_model = EfficientNetV2SWithDropout(num_classes=len(class_names))
        eff_model.load_state_dict(torch.load(eff_path, map_location="cpu"))
        eff_model.eval()
        models_dict['EfficientNetV2S'] = eff_model
    except Exception as e:
        st.warning(f"⚠️ EfficientNetV2S load failed: {e}")

    try:
        mobilenet_path = "mobilenetv3_state_dict.pth"
        download_if_needed("1R-UDjDkASZ277O4Ds7qyis4sTkmoLdwL", mobilenet_path)
        mobilenet_model = get_mobilenet_v3(num_classes=len(class_names))
        mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location="cpu"))
        mobilenet_model.eval()
        models_dict["MobileNetV3-Large"] = mobilenet_model
    except Exception as e:
        st.warning(f"⚠️ MobileNetV3-Large load failed: {e}")

    return models_dict

# Load
models = load_models()

# App UI
st.markdown("<h1 style='color:#198754;'>CassavAI-PPDVision</h1>", unsafe_allow_html=True)
st.subheader("Blending AI with Visual Diagnosis for Cassava PPD")
st.markdown("Upload a cassava tuber image and choose a model to predict the PPD score.")

# Model selection
model_choice = st.radio("**Select Model**", list(models.keys()) + ["Ensemble"], horizontal=True)

# File uploader
uploaded_file = st.file_uploader("Choose a cassava tuber image...", type=["jpg", "jpeg", "png"])

# Sidebar examples
st.sidebar.markdown("### Class Examples")
example_folder = "examples"
for class_name in class_names:
    class_path = os.path.join(example_folder, class_name)
    if os.path.isdir(class_path):
        st.sidebar.markdown(f"**{class_name.upper()}**")
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for img_file in image_files[:2]:
            img_path = os.path.join(class_path, img_file)
            st.sidebar.image(img_path, use_container_width=False, width=160)
    else:
        st.sidebar.warning(f"No folder for '{class_name}'")

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", width=300)
    temp_path = "temp.jpg"
    image.save(temp_path)

    try:
        if model_choice == 'Ensemble':
            result = predict_ensemble(
                temp_path,
                models['ResNet50'],
                models['EfficientNetV2S'],
                models['MobileNetV3-Large'],
                models['YOLOv8'],
                return_probs=True,
                verbose=False
            )
            if result is not None and result[0] is not None:
                top1_class_index, _, log_output = result
                predicted_class = class_names[top1_class_index]
                st.success(f"✅ Final Ensemble Prediction: **{predicted_class.upper()}**")
                st.markdown("### 📋 Ensemble Prediction Breakdown")
                st.text(log_output)
            else:
                st.error("❌ Ensemble prediction failed.")

        elif model_choice == 'YOLOv8':
            results = models['YOLOv8'](temp_path)
            top1_class_index = results[0].probs.top1
            predicted_class = class_names[top1_class_index]
            st.success(f"✅ Predicted Class ({model_choice}): **{predicted_class.upper()}**")

        else:
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = models[model_choice](img_tensor)
                top1_class_index = output.argmax().item()
                predicted_class = class_names[top1_class_index]
                st.success(f"✅ Predicted Class ({model_choice}): **{predicted_class.upper()}**")

    except Exception as e:
        st.error(f"❌ Prediction failed with {model_choice}: {e}")
