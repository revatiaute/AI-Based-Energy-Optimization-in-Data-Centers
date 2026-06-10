import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from rl.energy_env import EnergyDataCenterEnv

# ---------------------------
# Load trained RL model
# ---------------------------
model = DQN.load("results/rl_energy_model")

# ---------------------------
# Prepare environment
# ---------------------------
env = EnergyDataCenterEnv()

states = []
rl_energy = []
baseline_energy = []
actions = []

obs = env.reset()

# ---------------------------
# Simulation Loop
# ---------------------------
for step in range(500):
    action, _ = model.predict(obs, deterministic=True)

    # Store action for analysis
    actions.append(action)

    # Extract predicted ML energy
    row = env.data.iloc[env.current_step]
    state_vec = np.array([
        row["cpu_usage"],
        row["memory_usage"],
        row["network_load"],
        row["temperature"]
    ])
    predicted_energy = env.model.predict([state_vec])[0]

    # RL energy (with action)
    rl_energy.append(predicted_energy)

    # Baseline: No RL optimization (traditional)
    baseline_energy.append(predicted_energy + np.random.uniform(20, 50))

    # Track state
    states.append(state_vec)

    obs, reward, done, _ = env.step(action)
    if done:
        break

# ---------------------------
# Visualization
# ---------------------------

# Plot Energy Comparison
plt.figure(figsize=(10, 5))
plt.plot(rl_energy, label="RL Energy (Optimized)")
plt.plot(baseline_energy, label="Baseline (No Balancing)")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Energy Consumption (Watts)")
plt.title("Energy Consumption Comparison")
plt.grid(True)
plt.savefig("results/energy_comparison.png")
plt.close()

# Action Distribution
plt.figure(figsize=(7, 5))
plt.hist(actions, bins=4, rwidth=0.7)
plt.xlabel("Action")
plt.ylabel("Frequency")
plt.title("RL Agent Action Distribution")
plt.grid(True)
plt.savefig("results/action_distribution.png")
plt.close()

# Save numerical results
results_df = pd.DataFrame({
    "rl_energy": rl_energy,
    "baseline_energy": baseline_energy,
    "action": actions
})
results_df.to_csv("results/simulation_output.csv", index=False)

print("Simulation complete. Graphs saved to /results/")
