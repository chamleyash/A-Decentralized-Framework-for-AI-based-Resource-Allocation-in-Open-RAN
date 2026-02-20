import torch
import numpy as np
import torch.nn as nn


class LocalTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train(self, data):

        if not data:
            return 0.0

        self.model.train()

        states = []
        actions = []

        for sample in data:
            state = sample[0]
            action = sample[1]   # assuming second element is action index
            states.append(state)
            actions.append(action)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        self.optimizer.zero_grad()

        outputs = self.model(states)   # shape: [batch, action_dim]

        loss = self.criterion(outputs, actions)

        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def get_model_weights(self):
        return self.model.state_dict()