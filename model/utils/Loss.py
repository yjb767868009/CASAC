import torch


def mask_loss(x, y, data_length):
    """
    Calculate the loss of the square difference of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    mask = torch.zeros_like(x).float()
    for i in range(len(mask)):
        mask[i][:data_length[i]] = 1
    x = x * mask
    loss = torch.mean(torch.pow((x - y), 2))
    return loss


def mask_last_loss(x, y, data_length):
    """
    Calculate the loss of the square difference of the last digit of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    batch_size = x.size(0)
    all_loss = torch.zeros(1)
    if torch.cuda.is_available():
        all_loss = all_loss.cuda()
    for i in range(batch_size):
        loss = torch.mean(torch.pow((x[i, data_length[i] - 1, :] - y[i, data_length[i] - 1, :]), 2))
        all_loss += loss
    loss = all_loss / batch_size
    return loss
