import argparse

from model.bert.model import Model

# Check GPU available
from model.utils.initialization import initialization

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--cache", help="cache: if set as TRUE all the training data will be loaded at once"
                                    " before the training start.", action="store_true")
parser.add_argument("--epoch", default=200, type=int, help="data_root")
parser.add_argument("--batch_size", default=1, type=int, help="data_root")
parser.add_argument("--data_root", default="", help="data_root")
parser.add_argument("--save_path", default="", help="save model param path")
parser.add_argument("--load_path", default="", help="load model param path")
parser.add_argument("--train", help="train model", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    model = initialization(args)
    if args.train:
        model.train()
    if args.test:
        model.test(args.load_path if not args.train else "")
