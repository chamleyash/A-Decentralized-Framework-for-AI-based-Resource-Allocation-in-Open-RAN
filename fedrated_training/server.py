import copy
import torch

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights):
        avg = copy.deepcopy(client_weights[0])
        for key in avg:
            for i in range(1, len(client_weights)):
                avg[key] += client_weights[i][key]
            avg[key] = torch.div(avg[key], len(client_weights))
        self.global_model.load_state_dict(avg)
        return avg
