import copy
import torch

def median_average_models(local_models):
    """Coordinate-wise median aggregation."""
    if not local_models:
        raise ValueError("No models given")

    global_model = copy.deepcopy(local_models[0])

    for key in global_model.keys():
        stacked = torch.stack([m[key] for m in local_models], dim=0)
        median_vals = torch.median(stacked, dim=0).values
        global_model[key] = median_vals

    return global_model
