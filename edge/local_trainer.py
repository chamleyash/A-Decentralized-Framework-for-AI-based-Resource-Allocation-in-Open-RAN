class LocalTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, data):
        # placeholder for local training loop
        pass

    def get_model_weights(self):
        return self.model.state_dict()
