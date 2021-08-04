import os
import torch.utils.data as tordata

import argparse

from model.bert.model import Model
from model.utils.data_set import DataManager

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--cache", help="cache: if set as TRUE all the training data will be loaded at once"
                                    " before the training start.", action="store_true")
parser.add_argument("--epoch", default=200, type=int, help="data_root")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--data_len", default=10, type=int, help="data seq length")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--data_root", default="", help="data_root")
parser.add_argument("--save_path", default="", help="save model param path")
parser.add_argument("--load_path", default="", help="load model param path")
parser.add_argument("--train", help="train model", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
parser.add_argument("--view_attention", help="view attention", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    model = Model(args.save_path, args.lr)
    if args.train:
        train_data_manager = DataManager(os.path.join(args.data_root, "Train"), args.batch_size, data_len=args.data_len)
        test_data_manager = DataManager(os.path.join(args.data_root, "Test"), args.batch_size, data_len=args.data_len)
        model.train(train_data_manager, test_data_manager, args.epoch, args.load_path)
    if args.test:
        data_manager = DataManager(os.path.join(args.data_root, "Train"), args.batch_size, data_len=args.data_len)
        model.test(data_manager, args.load_path if not args.train else "")
    if args.view_attention:
        data_manager = DataManager(os.path.join(args.data_root, "Train"), args.batch_size, data_len=args.data_len)
        model.view_attention(data_manager, args.load_path)
