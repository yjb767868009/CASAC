conf = {
    'lr': 0.0001,
    'segmentation': [0, 419, 575, 2609, 4657, 5307],

    'encoder_nums': 5,
    'encoder_dims': [[419, 512, 256],
                     [156, 128, 128],
                     [2034, 1024, 256],
                     [2048, 1024, 256],
                     [650, 512, 256]],
    'encoder_activations': [['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu']],
    'encoder_dropout': 0.1,

    'key_bert_hidden': 1152,
    'motion_bert_hidden': 1296,
    'n_layers': 8,
    'attn_heads': 8,
    'bert_dropout': 0.1,
}
