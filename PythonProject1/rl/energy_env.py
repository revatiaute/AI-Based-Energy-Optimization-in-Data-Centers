import gymnasium as gym
import numpy as np
import pandas as pd

class EnergyDataCenterEnv(gym.Env):

    def __init__(self):
        super(EnergyDataCenterEnv, self).__init__()

        # Load dataset
        df = pd.read_csv("../data/workload.csv")

        self.cpu = df["cpu_usage"].values
        self.memory = df["memory_usage"].values
        self.network = df["network_load"].values
        self.temperature = df["temperature"].values
        self.energy = df["energy_consumption"].values

        self.index = 0
        self.max_index = len(df) - 1

        # Observation space = state (4 features)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        # 3 possible actions
        # 0 → keep as is
        # 1 → migrate load
        # 2 → lower frequency (DVFS)
        self.action_space = gym.spaces.Discrete(3)

    def _get_state(self):
        return np.array([
            self.cpu[self.index],
            self.memory[self.index],
            self.network[self.index],
            self.temperature[self.index]
        ], dtype=np.float32)

    def step(self, action):
        current_energy = self.energy[self.index]

        # Action impact on energy
        if action == 1:  # migrate load
            reward = -current_energy * 0.75  # save 25%
        elif action == 2:  # DVFS scaling
            reward = -current_energy * 0.70  # save 30%
        else:  # do nothing
            reward = -current_energy

        self.index += 1
        done = self.index >= self.max_index

        return self._get_state(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.index = 0
        return self._get_state(), {}
