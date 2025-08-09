"""
Medical example usage of M-TRUST with a simulated chest X-ray model.
"""

from mtrust import MTrustWrapper
import numpy as np

# Simulated chest X-ray classification model
class ChestXRayModel:
    def __init__(self):
        # Let's say our model predicts 5 diseases: Pneumonia, TB, COVID-19, Emphysema, Fibrosis
        self.diseases = ["Pneumonia", "Tuberculosis", "COVID-19", "Emphysema", "Fibrosis"]

    def predict(self, image):
        # This is where you’d run inference on your actual trained model
        # Here we simulate with random probabilities
        return np.random.rand(len(self.diseases))

# Create the model instance
original_model = ChestXRayModel()

# Wrap with M-TRUST for bias detection and mitigation
fair_model = MTrustWrapper(original_model, bias_threshold=0.8)

# Simulated patient chest X-ray data
image = np.random.rand(224, 224, 3)  # Replace with real preprocessed image

# Example patient demographics
demographics = {
    'age': 72,
    'gender': 'M',
    'race': 'Asian'
}

# Get predictions with bias checking
result = fair_model.predict(image, demographics, return_bias_info=True)

print("=== Prediction Results ===")
for disease, prob in zip(original_model.diseases, result['prediction']):
    print(f"{disease}: {prob:.3f}")

print("\nBias detected:", result['bias_detected'])
print("Mitigation applied:", result['mitigation_applied'])

# Generate a bias report
bias_report = fair_model.get_bias_report()
print("\n=== Bias Report ===")
print(bias_report)
