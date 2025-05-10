import logging
import random

import numpy as np
import torch
import yaml
from dotmap import DotMap
from rich.logging import RichHandler


def load_config(config_file: str = "src/config/config.yaml") -> DotMap:
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


def print_memory_usage():
    """
    Print detailed GPU memory usage, including PyTorch and system-level statistics
    """
    print("\n=== GPU Memory Usage ===")

    # PyTorch memory statistics
    print("\nPyTorch Memory Statistics:")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Reserved Memory: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  Peak Memory: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")

    # System-level memory statistics
    import subprocess

    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("\nSystem-level Memory Statistics (nvidia-smi):")
        print(nvidia_smi)
    except:
        print("\nFailed to get nvidia-smi output")

    print("=" * 50)


def setup_logging(log_dir=None, level=logging.INFO):
    handlers = [RichHandler(show_time=False, show_path=False)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "training.log"))

    logging.basicConfig(
        level=level,
        format=("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] \n%(message)s"),
        handlers=handlers,
    )
