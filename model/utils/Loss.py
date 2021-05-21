import torch


def base_loss(x, y):
    """
    Calculate the loss of the square difference of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    loss = torch.mean(torch.pow((x - y), 2))
    return loss


def last_loss(x, y):
    """
    Calculate the loss of the square difference of the last digit of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    loss = torch.mean(torch.pow((x[:, -1, :] - y[:, -1, :]), 2))
    return loss
