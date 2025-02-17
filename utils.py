import torch.nn as nn
import torch.nn.functional as F


def create_mlp(input_size, hidden_sizes, flatten_dim=None):
    layers = []
    if flatten_dim is not None:
        layers.append(nn.Flatten(flatten_dim))
    for i, size in enumerate(hidden_sizes):
        if i == 0:
            layers.append(
                nn.Linear(input_size, size))
        else:
            layers.append(nn.Linear(hidden_sizes[i - 1], size))

        if i < len(hidden_sizes) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def standardize_glimpse_locs(glimpse_locs):
    time_steps = glimpse_locs.shape[1]
    return F.layer_norm(glimpse_locs.permute(0, 2, 1), [time_steps]).permute(0, 2, 1)

