config = {
    'T0': { # Half output chanel size
        'nhid': [32, 16, 16, 8, 8, 16, 16, 32],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [32, 16, 16,  8, 8, 4,  4, 2],
        'input_scaling': [1]*8,
        'Paper_name': 'T0'
    },
    'T1': { # Half output chanel size
        'nhid': [32, 16, 16, 8, 8, 16, 16, 32],
        'sdil': [1,  2,   2,  4,  4,  6,  6, 8],
        'skrn': [64, 32, 32,  16, 16, 8,  8, 4],
        'input_scaling': [1]*8,
        'Paper_name': 'T1'
    }
}
