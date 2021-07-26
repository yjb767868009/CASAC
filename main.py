import os

import argparse

from model.bert.model import Model
from model.utils.data_set import load_data

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--cache", help="cache: if set as TRUE all the training data will be loaded at once"
                                    " before the training start.", action="store_true")
parser.add_argument("--epoch", default=200, type=int, help="data_root")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--data_root", default="", help="data_root")
parser.add_argument("--save_path", default="", help="save model param path")
parser.add_argument("--load_path", default="", help="load model param path")
parser.add_argument("--train", help="train model", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    model = Model(args.save_path, args.epoch, args.batch_size, args.lr)
    if args.train:
        train_source = load_data(os.path.join(args.data_root, "Train"), cache=args.cache)
        test_source = load_data(os.path.join(args.data_root, "Test"), cache=args.cache)
        model.train(train_source, test_source)
    if args.test:
        test_source = load_data(os.path.join(args.data_root, "Test"), cache=args.cache)
        model.test(test_source, args.load_path if not args.train else "")
