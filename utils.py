import random
import numpy as np
import torch
import yaml
from dotmap import DotMap


def load_config(config_file: str = "config/config.yaml") -> DotMap:
    """
    Loads configuration from the specified YAML file
    and returns a DotMap object for dot-notation access.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return DotMap(config_data)


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_model_memory(model):
    """Apply memory optimizations like proper gradient checkpointing setup"""
    # Ensure model is in training mode
    model.train()

    # Disable caching for gradient checkpointing
    model.config.use_cache = False

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Enable input gradients properly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model
