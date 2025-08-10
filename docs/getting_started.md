# Getting Started with M-TRUST

This guide walks you through:
1. A **Quick Start** example  
2. A **Full Model Example** using a trained DenseNet (CheXNet)  

---

## 🚀 Quick Start

```python
from mtrust import MTrustWrapper

class DummyModel:
    def predict(self, x): return [0.1, 0.2, 0.7]

fair_model = MTrustWrapper(DummyModel())
print(fair_model.predict("any_image.png", demographics={'race': 'Black'}))
