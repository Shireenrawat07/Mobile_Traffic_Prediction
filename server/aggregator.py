# server/aggregator.py
import copy

def average_models(local_models, weights=None):
    """
    Weighted averaging of model parameters.
    local_models: list of state_dicts
    weights: list of dataset sizes (same length as local_models)
    """
    if not local_models:
        raise ValueError("No local models provided for aggregation")

    global_model = copy.deepcopy(local_models[0])
    for key in global_model.keys():
        global_model[key] = 0.0

    if weights is None:
        weights = [1] * len(local_models)

    total_weight = sum(weights)

    for w, state in zip(weights, local_models):
        for key in state.keys():
            global_model[key] += (state[key] * w / total_weight)

    return global_model
