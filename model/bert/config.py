conf = {
    "embedding": {
        'segmentation': [0, 419, 575, 2609, 4657, 5307],
        'encoder_nums': 5,
        'encoder_dims': [[419, 512, 512],
                         [156, 128, 128],
                         [2034, 1024, 512],
                         [2048, 1024, 512],
                         [650, 512, 512]],
        'encoder_activations': [['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu']],
        'encoder_dropout': 0.3,
    },
    'hidden_dim': 512 + 128 + 512 + 512 + 512,
    'bert_dropout': 0.1,
}
