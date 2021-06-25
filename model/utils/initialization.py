import os
from model.utils import load_data
from model.bert.model import Model
from model.bert.config import conf


def initialization(args):
    print("Initializing model...")
    if args.train:
        train_source = load_data(os.path.join(args.data_root, "Train"), cache=args.cache)
        test_source = load_data(os.path.join(args.data_root, "Test"), cache=args.cache)
    elif args.test:
        train_source = load_data(args.data_root, cache=args.cache)
        test_source = train_source
    else:
        train_source = None
        test_source = None
    model = Model(train_source, test_source, args.save_path, args.epoch, args.batch_size, args.lr)
    print("Model initialization complete.")
    return model
