import numpy as np

class RANEnvironment:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2

    def reset(self):
        self.state = np.random.rand(self.state_dim)
        return self.state

    def step(self, action):
        bandwidth = np.clip(action[0], 0, 1)
        power = np.clip(action[1], 0, 1)

        # Simulated metrics
        throughput = bandwidth * 10        # Mbps
        latency = (1 - power) * 100         # ms

        reward = throughput - 0.1 * latency
        next_state = np.random.rand(self.state_dim)

        return next_state, reward, throughput, latency
