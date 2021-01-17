import torch
import torch.nn as nn

from model.network.Discriminative import Discriminative
from model.network.Encoder import Encoder
from model.network.Refiner import Refiner
from model.network.RNN import RNN


def build_network(name, dims, activations, dropout):
    network = eval(name)(dims, activations, dropout)
    if torch.cuda.is_available():
        network.cuda()
    network = nn.DataParallel(network)
    return network
