import argparse

from model.bert.model import Model
from model.utils.data_preprocess import data_preprocess

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
parser.add_argument("--pretrain", help="retrain model", action="store_true")
parser.add_argument("--key_train", help="train key prediction model", action="store_true")
parser.add_argument("--motion_train", help="train motion prediction model", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    train_flag = args.key_train or args.motion_train
    model = initialization(epoch=args.epoch, batch_size=args.batch_size, data_root=args.data_root,
                           save_path=args.save_path, load_path=args.load_path, cache=args.cache,
                           train=train_flag)
    if train_flag:
        model.train(args.pretrain, args.key_train, args.motion_train)
    if args.test:
        model.test(args.load_path if not train_flag else "")
