"""
Basic usage example of M-TRUST
"""

from mtrust import MTrustWrapper
import numpy as np

# Your existing medical AI model
class YourMedicalModel:
    def predict(self, image):
        # Your model's prediction logic
        return np.random.rand(14)  # Example: 14 disease probabilities

# Wrap with M-TRUST
model = YourMedicalModel()
fair_model = MTrustWrapper(model, bias_threshold=0.85)

# Make fair predictions
image = np.random.rand(224, 224, 3)  # Example image
demographics = {
    'age': 65,
    'gender': 'F',
    'race': 'Black'
}

result = fair_model.predict(image, demographics, return_bias_info=True)

print(f"Prediction: {result['prediction']}")
print(f"Bias detected: {result['bias_detected']}")
print(f"Mitigation applied: {result['mitigation_applied']}")

# Get bias report
report = fair_model.get_bias_report()
print(f"Bias report: {report}")