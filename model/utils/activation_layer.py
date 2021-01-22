import torch.nn as nn

activation_layer_list = {
    'Relu': nn.ReLU( ),
    'elu': nn.ELU( ),
    'softmax': nn.Softmax(dim=1),
    'Sigmoid': nn.Sigmoid(),
    'None': None
}


def activation_layer(s):
    return activation_layer_list.get(s)


def build_layer(layer_name, ):
    layer = nn.Linear()
    return layer
