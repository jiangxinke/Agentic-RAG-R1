
class NeuronMetric:
    def __init__(self):
        self.num_samples = 0
        self.metrics = {}

        self.keys = set()
        self.layer_indices = set()

    def get(self, key, layer_idx):
        return self.metrics.get(key, {}).get(layer_idx, 0)

    def update(self, key, layer_idx, value, n=1):
        if key not in self.metrics:
            self.keys.add(key)
            self.metrics[key] = {}
        if layer_idx not in self.metrics[key]:
            self.layer_indices.add(layer_idx)
            self.metrics[key][layer_idx] = value
        else:
            self.metrics[key][layer_idx] += value
        self.num_samples += n

    def compute_average(self):
        avg_metrics = {}
        for key, layer_dict in self.metrics.items():
            avg_metrics[key] = {}
            for layer_idx, values in layer_dict.items():
                avg_metrics[key][layer_idx] = values / self.num_samples
        return avg_metrics