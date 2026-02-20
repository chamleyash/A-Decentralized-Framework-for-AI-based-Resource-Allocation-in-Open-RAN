import numpy as np

class NetworkEnvironment:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = self.reset()

    def reset(self):
        self.state = np.random.rand(self.state_dim)
        return self.state

    def step(self, action):
        # Placeholder dynamics
        next_state = np.random.rand(self.state_dim)
        reward = self._compute_reward(self.state, action, next_state)
        self.state = next_state
        return next_state, reward

    def _compute_reward(self, state, action, next_state):
        return float(np.random.rand())
