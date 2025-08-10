# M-TRUST: Multimodal Trustworthy Healthcare AI

[![PyPI version](https://badge.fury.io/py/mtrust-medical.svg)](https://badge.fury.io/py/mtrust-medical)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🏥 Make ANY Medical AI Model Fair and Trustworthy

M-TRUST is a bias detection and mitigation framework that wraps any medical AI model to ensure fairness across patient demographics.

## ✨ Features

- 🎯 **4 Types of Bias Detection**: Demographic, Quality, Annotation, Amplification
- 🛡️ **Real-time Bias Mitigation**: Automatic fairness adjustments
- 📊 **Fairness Metrics**: Track and report bias metrics
- 🔧 **Easy Integration**: One line to wrap any model
- 🏥 **Medical-Specific**: Designed for healthcare AI

## 📦 Installation

```bash
pip install mtrust-medical

## 🚀 Usage Example

```python
# ==========================
#  M-TRUST Quick & Full Demo
# ==========================
from mtrust import MTrustWrapper

# ---------- QUICK CHECK ----------
# Dummy model for fast verification
class DummyModel:
    def predict(self, x): return [0.1, 0.2, 0.7]

print("\n=== QUICK CHECK ===")
fair_dummy = MTrustWrapper(DummyModel())
result = fair_dummy.predict("any_image.png", demographics={'race': 'Black'})
print("Quick check result:", result)

# ---------- REAL MODEL EXAMPLE ----------
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

print("\n=== REAL MODEL USAGE ===")

# 1. Load model
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(num_ftrs, 14), nn.Sigmoid())

# 2. Load weights (replace path with your .pth file)
ckpt = torch.load("models/chexnet_model.pth", map_location="cpu")
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(x).squeeze().cpu().numpy()

# 4. Wrap model with M-TRUST
fair_model = MTrustWrapper(
    original_model=type('WrappedModel', (), {'predict': predict})(),
    bias_threshold=0.85,
    mitigation_strength='balanced'
)

# 5. Predict & report bias
result = fair_model.predict("data/sample_xray.png",
                            demographics={'age': 65, 'gender': 'F', 'race': 'Asian'})
print("Prediction:", result)
print("Bias report:", fair_model.get_bias_report())

