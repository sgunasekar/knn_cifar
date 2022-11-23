import os
import torch
import numpy as np
import faiss

mu = np.array([0.4914, 0.4822, 0.4465])
sigma = np.array([0.2470, 0.2435, 0.2616])

im_dim = 32
d = im_dim*im_dim*3


## AA config
pad = 4
num_epochs = 1600
aa_config_string = 'rand-m5-mstd0.5-inc1'
aa_params = dict(translate_const=int(im_dim * 0.45), img_mean=tuple([min(255, round(255 * channel_mean)) for channel_mean in mu]))
reprob = 0.25
mixup_alpha = 0.8
cutmix_alpha = 1.0
cutmix_minmax = None
mixup_prob = 1
mixup_switch_prob = 0.5
mixup_mode = 'batch'
smoothing = 0.1

default_cfg={
    #"gpu": True,
    "dataset": "CIFAR10",
    "root": os.path.join("..","data","CIFAR10"),
    "save_dir": os.path.join("..","save","ANN"),
    "epoch_size": 50000,
    "test_size": 10000,
    "d": 32*32*3,
    "num_classes": 10,
    "index_map": dict(exact = faiss.IndexFlatL2, pq = faiss.IndexPQ, ivfpq = faiss.IndexIVFPQ),
    "index_args_map": dict(exact = (d,), pq = (d,32,8), ivfpq = (faiss.IndexFlatL2(d), d, 10, 32, 8)),
    "index_suffix_map": dict(exact = '', pq = '_32_8', ivfpq = '_10_32_8'),
    "Mixup_kwargs": dict(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=cutmix_minmax, prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode, label_smoothing=smoothing)
}

def parser_add_arguments(parser):
    ## Data augmentation args
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--basic-augmentation', action='store_true',  dest='basic_augmentation', help='flag to use basic augmentation (default: True)')
    parser.add_argument('--no-basic-augmentation', default=True, action='store_false', dest='basic_augmentation',help='flag to not use basic augmentation')
    parser.set_defaults(basic_augmentation=False)
    group1.add_argument('--advanced-augmentation', '--adv-aug', default=False, action='store_true',  help='flag to use auto augmentation (default: False)')
    
    parser.add_argument('--epochs', default=num_epochs, type=int, help="no. of epochs equivalent of augmented data to use for ANN classifier -- used only for advanced augmentation")

    parser.add_argument('--train-aug', default=0, type = int,  help='in advanced augmentation, this argument specified the no. of augmentations of each training datapoint to use in training of the ann_index (not the ann_classifier itself, which uses `--epochs` numbers of augmenations)')

    parser.add_argument('--use-mixup', default=False, action='store_true',  help='flag to use mixup/cutmix  (default: False)')

    parser.add_argument('--pca', default=0, type = int, help='number of pca components to use (default: 0=no-pca)')

    parser.add_argument('--indexes', default=['ivfpq'], nargs='+', help="use from ['exact','pq,'ivfpq','sq8','sq16']")

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for training and testing')

    return parser