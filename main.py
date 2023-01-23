# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import importlib
import sys
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed


def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    print(args)

    if args.load_best_args:
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)

        # Up to this point, the parsed arguments are only basic ones and used for get the best args
        args = parser.parse_known_args()[0]

        if args.model == 'joint':
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset]['sgd'][args.weight_dist]
            else:
                best = best_args[args.dataset]['sgd']
        else:
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset][args.model][args.weight_dist]
            else:
                best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
            
        # for key, value in best.items():
        #     setattr(args, key, value)

        # Get the parser from the specific model and dataset
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)

        # Modified the input arguments according to best_args
        to_parse = sys.argv[1:]

        to_parse = to_parse + ['--' + k + '=' + str(v) 
                               for k, v in best.items() 
                               if '--' + k not in to_parse]
                               
        to_parse.remove('--load_best_args')
        
        # Parse the arguments
        args = parser.parse_args(to_parse)

    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)
        args = parser.parse_args()

        print(args)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task')
        ctrain(args)


if __name__ == '__main__':
    main()
