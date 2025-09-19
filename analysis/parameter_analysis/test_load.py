import torch

save_path = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/neuron_importance.pt"
neuron_importance = torch.load(save_path)
print(neuron_importance)