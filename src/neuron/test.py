import torch
from collections import defaultdict

model: torch.nn.modules
neuron_importance_dict = defaultdict(0)
for name, param in model.named_parameters():
    def metric_fn(param):
        pass
    param_value = metric_fn(param).detach()
    neuron_importance_dict[name] += param_value
