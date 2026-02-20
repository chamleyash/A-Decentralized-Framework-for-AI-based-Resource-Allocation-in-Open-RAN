from regional.aggregation_logic import fedavg

class GlobalServer:
    def __init__(self):
        self.regional_updates = []
        self.global_model = None

    def receive_update(self, regional_weights):
        self.regional_updates.append(regional_weights)

    def aggregate(self):
        if not self.regional_updates:
            return None

        self.global_model = fedavg(self.regional_updates)
        self.regional_updates = []
        return self.global_model
