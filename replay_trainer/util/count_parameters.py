import torch
import torch.nn as nn

def count_parameters(model: nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    
    # Convert to human-readable format
    if total_params >= 1e9:
        return f'{total_params/1e9:.2f}B parameters'
    elif total_params >= 1e6:
        return f'{total_params/1e6:.2f}M parameters'
    elif total_params >= 1e3:
        return f'{total_params/1e3:.2f}K parameters'
    else:
        return f'{total_params} parameters'