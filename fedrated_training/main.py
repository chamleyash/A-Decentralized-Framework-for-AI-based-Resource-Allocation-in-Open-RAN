import csv
import numpy as np
from edge import EdgeAgent
from server import FederatedServer
from model import DQN

NUM_AGENTS = 3
ROUNDS = 5

agents = [EdgeAgent(i) for i in range(NUM_AGENTS)]
server = FederatedServer(DQN())

log_data = []

for r in range(ROUNDS):
    print(f"\n--- Round {r+1} ---")
    weights = []

    for agent in agents:
        agent.train_local()
        weights.append(agent.get_weights())
        log_data.extend(agent.logs)

    global_weights = server.aggregate(weights)

    for agent in agents:
        agent.set_weights(global_weights)

# Save logs
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Reward", "Throughput", "Latency"])
    writer.writerows(log_data)

print("Training completed & metrics saved.")
