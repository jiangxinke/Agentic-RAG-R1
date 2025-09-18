import torch

save_path = "output_eval/neuron/neuron_importance.pt"
neuron_importance = torch.load(save_path)
print(neuron_importance)