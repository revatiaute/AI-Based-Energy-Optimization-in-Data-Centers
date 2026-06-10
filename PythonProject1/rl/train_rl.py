import gymnasium as gym
from stable_baselines3 import DQN
import os
from energy_env import EnergyDataCenterEnv

# --------------------------------------
# Create environment
# --------------------------------------
env = EnergyDataCenterEnv()

# --------------------------------------
# Create RL logs folder
# --------------------------------------
os.makedirs("../results/rl_logs", exist_ok=True)

# --------------------------------------
# Create DQN agent
# --------------------------------------
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    gamma=0.95,
    buffer_size=50000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1
)

# --------------------------------------
# Train RL agent
# --------------------------------------
print("Training RL agent... please wait")
model.learn(total_timesteps=5000)
print("Training completed!")

# --------------------------------------
# Save the model
# --------------------------------------
model.save("../results/rl_dqn_model")
print("RL model saved successfully!")
