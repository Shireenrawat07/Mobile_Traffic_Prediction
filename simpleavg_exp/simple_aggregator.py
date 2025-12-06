# simple_aggregator.py

import torch

def simple_average_models(local_models):
    """
    Simple Average Aggregator
    - Each client contributes equally
    - No weighting by dataset size
    - Average = (model1 + model2 + ... + modelN) / N
    """
    if not local_models:
        raise ValueError("No local models provided for simple average aggregation")

    # Initialize empty dictionary
    avg_state = {}

    # Get model parameter keys
    keys = local_models[0].keys()

    # Compute mean for every layer
    for key in keys:
        # Stack all client tensors for this layer
        stacked = torch.stack([state[key] for state in local_models], dim=0)

        # Simple unweighted average
        avg_state[key] = torch.mean(stacked, dim=0)

    return avg_state
