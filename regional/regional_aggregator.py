from duckdb import aggregate
from regional.aggregation_logic import fedavg

class RegionalAggregator:
    def __init__(self, region_id):
        self.region_id = region_id
        self.edge_updates = []

    def receive_update(self, model_weights):
        self.edge_updates.append(model_weights)

    def aggregate(self):
        if not self.edge_updates:
            return None
        aggregated = fedavg(self.edge_updates)
        self.edge_updates = []
        return aggregated
