import argparse

import torch.utils
import torch.utils.cpp_extension

from model.bert.model import Model
from model.utils.data_preprocess import data_preprocess

# Check GPU available
from model.utils.initialization import initialization

if torch.cuda.is_available():
    print("CUDA_HOME:", torch.utils.cpp_extension.CUDA_HOME)
    print("torch cuda version:", torch.version.cuda)
    print("cuda is available:", torch.cuda.is_available())

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--cache", help="cache: if set as TRUE all the training data will be loaded at once"
                                    " before the training start.", action="store_true")
parser.add_argument("--data_preprocess", default="", help="if need preprocess dataset input and output")
parser.add_argument("--epoch", default=200, type=int, help="data_root")
parser.add_argument("--batch_size", default=1, type=int, help="data_root")
parser.add_argument("--data_root", default="", help="data_root")
parser.add_argument("--save_path", default="", help="save model param path")
parser.add_argument("--load_path", default="", help="load model param path")
parser.add_argument("--train", help="train generate model", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    if args.data_preprocess != "":
        data_preprocess(args.data_preprocess)
    model = initialization(args.epoch, args.batch_size, args.data_root, args.save_path, args.load_path, args.cache)
    if args.train is True:
        model.train()
    if args.test:
        model.test()
