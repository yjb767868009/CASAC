conf = {
    "embedding": {
        'segmentation': [0, 419, 575, 2609, 4657, 5307],
        'encoder_nums': 5,
        'encoder_dims': [[419, 512],
                         [156, 128],
                         [2034, 512],
                         [2048, 512],
                         [650, 512]],
        'encoder_activations': [['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu'],
                                ['elu', 'elu']],
        'encoder_dropout': 0.3,
    },
    'hidden_dim': 512 + 128 + 512 + 512 + 512,
    'bert_dropout': 0.1,
    'attention_head_nums': 17,
}
