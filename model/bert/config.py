conf = {
    'lr': 0.001,
    'segmentation': [0, 419, 575, 2609, 4657, 5307],

    'encoder_nums': 5,
    'encoder_dims': [[419, 256, 128],
                     [156, 256, 128],
                     [2034, 512, 128],
                     [2048, 512, 128],
                     [650, 256, 128]],
    'encoder_activations': [['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu'],
                            ['elu', 'elu']],
    'encoder_dropout': 0.3,

    'hidden': 640,
    'n_layers': 8,
    'attn_heads': 8,
    'bert_dropout': 0.1,
}
