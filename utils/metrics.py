class MetricsTracker:
    def __init__(self):
        self.metrics = {}

    def log(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get(self, key):
        return self.metrics.get(key, [])

    def summary(self):
        return {
            k: sum(v) / len(v) if len(v) > 0 else 0
            for k, v in self.metrics.items()
        }
