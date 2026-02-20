import numpy as np

class TrafficGenerator:
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def generate(self):
        return np.random.rand(self.state_dim)
