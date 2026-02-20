class RewardModel:
    def __init__(self, config):
        self.config = config

    def compute(self, state, action, next_state):
        raise NotImplementedError
