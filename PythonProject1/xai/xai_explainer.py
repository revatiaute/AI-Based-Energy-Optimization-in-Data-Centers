import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# Load trained ML energy model
model = joblib.load("../ml/energy_model.pkl")

# Load dataset
df = pd.read_csv("../data/workload.csv")

# Features used for ML prediction
features = ["cpu_usage", "memory_usage", "network_load", "temperature"]
X = df[features]

# Create SHAP Explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Create output folder for SHAP plots
os.makedirs("../results/shap_plots", exist_ok=True)

# 1️⃣ Global Feature Importance
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("../results/shap_plots/global_feature_importance.png")
plt.close()

# 2️⃣ Local explanation for first sample
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("../results/shap_plots/sample_explanation.png")
plt.close()

print("SHAP XAI explanations generated successfully!")
