import torch

def fedavg(model_weights_list):
    avg_weights = {}

    for key in model_weights_list[0].keys():
        avg_weights[key] = sum(
            w[key] for w in model_weights_list
        ) / len(model_weights_list)

    return avg_weights
