import torch
import datasets
import models
import training as tr
import numpy as np
import argparse
from operator import attrgetter
import os
import random
import time
import torch.nn as nn


def seed_everything(seed=127):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def fully_train(model, opt,dataset):
    dataset = getattr(datasets, dataset)()
    if opt.surrogate:
        model, acc=tr.train_regression(model, dataset,
                        '%s/model.pth' % (opt.savedir),  teacher=opt.teacher, surrogate = opt.surrogate)
    else:
        model, acc=tr.train_regression(model, dataset,
                        '%s/model.pth' % (opt.savedir))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for surrogate model')

    parser.add_argument('--network', type=str, default='groundtruth')
    parser.add_argument('--dataset', type=str, default='artificial')
    parser.add_argument('--suffix', type=str, default='0', help='0/1/2/3...')
    parser.add_argument('--teacher', type=str, default='save/groundtruth_Artificial_0/model.pth')
    parser.add_argument('--surrogate', action='store_true', default=False, help="using surrogate model")

    args = parser.parse_args()

    seed_everything()

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'artificial']

    if args.dataset == 'cifar10':
        args.dataset = 'CIFAR10Val'
    elif args.dataset == 'cifar100':
        args.dataset = 'CIFAR100Val'
    elif args.dataset == 'imagenet':
        args.dataset = 'IMAGENETVal'
    elif args.dataset == 'artificial':
        args.dataset = 'Artificial'


    args.savedir = './save/%s_%s_%s' % (args.network, args.dataset,
                                  args.suffix)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)


    model = getattr(models, args.network)()
    if args.surrogate:
        args.teacher = torch.load(args.teacher)
    print(model)
    fully_train(model, args, dataset=args.dataset)
