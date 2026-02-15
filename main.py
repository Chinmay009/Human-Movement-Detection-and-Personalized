import pickle
import torch
import numpy as np
from NN_model import HARNet
from config import USER_PROFILE
from config import THRESHOLDS
from llm import generate_advisory

# Load ML model and scaler
with open("nn_activity_model.pkl", "rb") as f:
    checkpoint = pickle.load(f)

model = HARNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
scaler = checkpoint["scaler"]

# Simulated input (replace with real ML prediction)
activity_info = {
    "Activity": "Seating",
    "Duration": 90,
    "Temperature": 30
}

X_new = np.random.rand(1, 561)
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

with torch.no_grad():
    output = model(X_new_tensor)
    probs = torch.softmax(output, dim=1)
    pred_id = torch.argmax(probs, axis=1).item()

activities = ["Walking", "Running", "Stepping", "Resting", "Sitting", "Lying"]
activity_info["Activity"] = activities[pred_id]

# Generate advisory from LLM
response = generate_advisory(USER_PROFILE, activity_info)
print(response)
