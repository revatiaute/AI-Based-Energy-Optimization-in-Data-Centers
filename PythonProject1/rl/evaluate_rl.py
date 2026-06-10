import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from energy_env import EnergyDataCenterEnv
import matplotlib.pyplot as plt
import os

# Load trained model
model = DQN.load("../results/rl_dqn_model")

# Create environment
env = EnergyDataCenterEnv()

states = []
baseline_energy = []
rl_energy = []

state, _ = env.reset()

done = False

print("Running RL evaluation...")

while not done:
    # RL action
    action, _ = model.predict(state, deterministic=True)

    # Current energy
    current_energy = env.energy[env.index]

    # Baseline = no action
    baseline_energy.append(current_energy)

    # RL Energy after action
    if action == 0:
        energy = current_energy
    elif action == 1:
        energy = current_energy * 0.75
    elif action == 2:
        energy = current_energy * 0.70

    rl_energy.append(energy)

    states.append(state)

    state, reward, done, _, _ = env.step(action)

# Convert to DataFrame
df = pd.DataFrame({
    "baseline_energy": baseline_energy,
    "rl_energy": rl_energy
})

os.makedirs("../results/rl_results", exist_ok=True)

df.to_csv("../results/rl_results/rl_energy_comparison.csv", index=False)

print("Evaluation completed! Results saved.")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(baseline_energy[:100], label="Baseline (No Load Balancing)")
plt.plot(rl_energy[:100], label="RL Optimized Energy")
plt.title("Baseline vs RL Energy Consumption (First 100 Samples)")
plt.xlabel("Time Step")
plt.ylabel("Energy Consumption (Watt)")
plt.legend()
plt.grid(True)
plt.savefig("../results/rl_results/energy_plot.png")
plt.show()

print("Energy graph saved!")
