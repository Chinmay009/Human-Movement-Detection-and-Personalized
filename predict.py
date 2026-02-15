import pickle
import torch
import numpy as np
from NN_model import HumanActivityNet  # Make sure this matches your NN class name

# Load the saved model and scaler
with open("nn_model.pkl", "rb") as f:
    checkpoint = pickle.load(f)

model = HumanActivityNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
scaler = checkpoint["scaler"]

# Example input: replace this with real sensor data
X_new = np.random.rand(1, 561)  # 1 sample, 561 features
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# Run inference
with torch.no_grad():
    output = model(X_new_tensor)
    probs = torch.softmax(output, dim=1)
    pred_id = torch.argmax(probs, axis=1).item()

activities = ["Walking", "Running", "Stepping", "Resting", "Sitting", "Lying"]
print("Predicted Activity:", activities[pred_id])
