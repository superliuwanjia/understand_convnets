import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()
parser.add_argument('-pb',
                    '--pb',
                    type=str2bool,
                    default=True,
                    help='init bias slightly positive to 0.1, 0 if turned off'
                    )
parser.add_argument('-dataset',
                    '--dataset',
                    type=str,
                    default='2Rec_64_4000_20_1_black',
                    help='specify the dataset to use'
                    )
parser.add_argument('-is_viz',
                    '--is_viz',
                    type=str2bool,
                    default=False,
                    help='true to turn on the visual block'
                    )
parser.add_argument('-std',
                    '--std',
                    type=float,
                    default=1e-1,
                    help='specify the init std for the training'
                    )
parser.add_argument('-gpus',
                    '--gpus',
                    type=str,
                    default='7',
                    help='specify which GPU to use'
                    )
parser.add_argument('-epochs',
                    '--epochs',
                    type=int,
                    default=100,
                    help='specify the total # of epochs for the training'
                    )
parser.add_argument('-lr',
                    '--lr',
                    type=float,
                    default=1e-3,
                    help='specify the learning rate for the training'
                    )
parser.add_argument('-bs',
                    '--bs',
                    type=int,
                    default=128,
                    help='specify the batch size for the training'
                    )
parser.add_argument('-p_accu',
                    '--p_accu',
                    type=int,
                    default=5,
                    help='specify for every how many steps print the accuracy')
parser.add_argument('-num_neurons',
                    '--num_neurons',
                    type=int,
                    default=100,
                    help='specify the # of hidden neurons for each layer'
                    )
parser.add_argument('-num_layers',
                    '--num_layers',
                    type=int,
                    default=1,
                    help='specify the # of hidden layers'
                    )
parser.add_argument('-pa',
                    '--pa',
                    type=int,
                    default=1250,  # 50 epochs
                    help='specify the patience which is used in early stop')
parser.add_argument('-rs',
                    '--rs',
                    type=int,
                    default=42,
                    help='specify the random seed for the training'
                    )
parser.add_argument('-s_path',
                    '--s_path',
                    type=str,
                    default='summaries',
                    help='specify the summary path for the training'
                    )


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed