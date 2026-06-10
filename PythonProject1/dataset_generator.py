import pandas as pd
import numpy as np
import os

# Make sure data directory exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)

time_steps = 1000

data = {
    "cpu_usage": np.random.uniform(10, 95, time_steps),
    "memory_usage": np.random.uniform(20, 90, time_steps),
    "network_load": np.random.uniform(5, 80, time_steps),
    "temperature": np.random.uniform(30, 85, time_steps),
    "energy_consumption": np.random.uniform(200, 800, time_steps)
}

df = pd.DataFrame(data)
df.to_csv("data/workload.csv", index=False)

print("Dataset generated successfully → data/workload.csv")
