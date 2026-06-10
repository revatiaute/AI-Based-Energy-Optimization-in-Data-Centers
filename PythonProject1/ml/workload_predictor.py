import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("../data/workload.csv")

features = ["cpu_usage", "memory_usage", "network_load", "temperature"]
target = "energy_consumption"   # ⭐ Changed Target

X = data[features]
y = data[target]

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Train ML Model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 4. Save Model
# -------------------------------
os.makedirs("../ml", exist_ok=True)
joblib.dump(model, "../ml/energy_model.pkl")

print("Energy Prediction Model Saved")

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nML Energy Model Performance")
print("---------------------------")
print(f"Mean Squared Error : {mse:.2f}")
print(f"R² Score           : {r2:.2f}")

# -------------------------------
# 7. Save Predictions
# -------------------------------
predicted_df = X_test.copy()
predicted_df["actual_energy"] = y_test.values
predicted_df["predicted_energy"] = y_pred

os.makedirs("../results", exist_ok=True)
predicted_df.to_csv("../results/energy_predictions.csv", index=False)

print("\nPredictions saved to results/energy_predictions.csv")

# -------------------------------
# 8. Visualization
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(y_test.values[:50], label="Actual Energy", marker='o')
plt.plot(y_pred[:50], label="Predicted Energy", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Energy Consumption (Watts)")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.grid(True)
plt.show()
