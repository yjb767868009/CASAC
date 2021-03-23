import torch
import torch.nn as nn
import torch.version
import torch.utils
import torch.utils.cpp_extension

if torch.cuda.is_available():
    print("cuda is available:", torch.cuda.is_available())
    print("CUDA_HOME:", torch.utils.cpp_extension.CUDA_HOME)
    print("torch cuda version:", torch.version.cuda)
    print("Using %d GPUS" % torch.cuda.device_count())
else:
    print("Using CPU")


def model2gpu(x):
    if torch.cuda.is_available():
        return nn.DataParallel(x.cuda())
    return x


def data2gpu(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x
