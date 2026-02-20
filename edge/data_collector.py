class DataCollector:
    def __init__(self):
        self.buffer = []

    def collect(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def get_data(self):
        return self.buffer

    def clear(self):
        self.buffer = []
