"""
This module contains MLP model.
"""
import torch.nn as nn
import numpy as np

class PriorNetMLP(nn.Module):
    """
    Simple MLP model with all fully connected layers of this form:
    len(fc_layers) * [FC_layer > ReLU > Dropout_layer]
    with a final output layer dictated by n_out.
    """
    def __init__(self, fc_layers, n_in=28, n_out=10, drop_rate=0.5):
        assert type(fc_layers) in [list, np.ndarray]
        assert len(fc_layers) > 0
        assert n_in > 0
        assert n_out > 0

        super(PriorNetMLP, self).__init__()
        layers = []
        prev_layer_size = n_in * n_in
        for layer in fc_layers:
            layers += [nn.Linear(prev_layer_size, layer), nn.ReLU(), nn.Dropout(p=drop_rate)]
            prev_layer_size = layer

        # output layers
        layers += [nn.Linear(fc_layers[-1], n_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1) # flatten the image
        logits = self.layers(inputs)
        return logits