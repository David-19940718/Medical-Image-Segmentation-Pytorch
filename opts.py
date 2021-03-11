from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import warnings
from distutils.version import LooseVersion

import torch
from torch import optim

from loss import *
from architectures import get_model
from utils.utils import setup_seed

warnings.filterwarnings("ignore")
assert sys.version >= '3.6.0', 'Python version>=3.6.0 is better.'
assert LooseVersion(torch.__version__) >= LooseVersion('1.3.0'), 'PyTorch version>=1.3.0 is required.'


class Opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Pytorch framework for 2D segmentation.")
        # Basic experiment setting
        self.parser.add_argument('-g', '--gpus', type=str, default='0',
                                 help='GPU id to use.')
        self.parser.add_argument('--exp_id', required=True, help='experiment name')
        self.parser.add_argument('--optim', default='Adam', help='optimizer alghorithm'
                                                                 'Adam | SGD | *')
        self.parser.add_argument('--sche', default='Poly', help='learning rate scheduler'
                                                                'Poly | ExpLR | MulStepLR | CosAnnLR | ReduceLR | *')
        self.parser.add_argument('--loss', default='dice_bce_loss', help='loss function.(default bce_loss)'
                                                                         'dice_loss | bce_loss | dice_bce_loss | *')
        self.parser.add_argument('--vis', action='store_true',
                                 help='Visualize the training process.[Default False]')
        self.parser.add_argument('-p', '--port', type=int, default=8098, help='random seed.[Default 8098]')
        self.parser.add_argument('--amp', action='store_false',
                                 help='Turn on automatic mixed precision training.[Default True]')
        # Model
        self.parser.add_argument('--height', type=int, default=512, help='height of image.')
        self.parser.add_argument('--width', type=int, default=512, help='width of image.')
        self.parser.add_argument('--n_channels', type=int, default=3, help='number of channels.')
        self.parser.add_argument('--n_classes', type=int, default=1, help='number of classes.')
        self.parser.add_argument('--l2', type=float, nargs='?', default=1e-8,
                                 help='L2 norm')
        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='UNet',
                                 help='model architecture (default: UNet)')
        # Train
        self.parser.add_argument('-s', '--seed', type=int, default=12345,
                                 help='Seed for initializing training.[Default 0]')
        self.parser.add_argument('-e', '--epochs', type=int, default=80,
                                 help='Number of total epochs to run')
        self.parser.add_argument('-j', '--num_workers', type=int, nargs='?', default=2, metavar='J',
                                 help='Number of data loading workers (default: 4)')
        self.parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=2, metavar='B',
                                 help='Batch size')
        self.parser.add_argument('-l', '--lr', type=float, nargs='?', default=1e-3, metavar='LR',
                                 help='Initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                                 help='weight decay (default: 1e-4)', dest='weight_decay')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume the experiments.[Default False]')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                                 help='Use multi-processing distributed training to launch '
                                      'N processes per node, which has N GPUs. This is the '
                                      'fastest way to use PyTorch for either single node or '
                                      'multi node data parallel training')
        self.parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
        self.parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        self.parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')

        # Dataset
        self.parser.add_argument('--dataset', required=True,
                                 help='please specify the dataset which you use.')
        self.parser.add_argument('--aug', action='store_true',
                                 help='Data augmentation.[Default False]')
        # Test
        self.parser.add_argument('--tta', action='store_true',
                                 help='Test Time Augmentation.[Default False]')
        self.parser.add_argument('--roc', action='store_true',
                                 help='Whether to save NumPy array to draw roc curve.[Default False]')
        self.parser.add_argument('--threshold', type=int, default=0,
                                 help='threshold of label for post-process.')

    def parse_arg(self):
        opt = self.parser.parse_args()
        setup_seed(opt.seed)
        opt.amp_available = True if LooseVersion(torch.__version__) >= LooseVersion('1.6.0') and opt.amp else False
        ####################################################################################################
        """ Directory """
        dir_root = os.getcwd()
        opt.dir_data = os.path.join(dir_root, 'data', opt.dataset)
        opt.dir_img = os.path.join(opt.dir_data, 'image')
        opt.dir_label = os.path.join(opt.dir_data, 'label')
        opt.dir_log = os.path.join(dir_root, 'logs', opt.dataset, f"EXP_{opt.exp_id}_NET_{opt.arch}")

        opt.dir_vis = os.path.join(dir_root, 'vis', opt.dataset, opt.exp_id)
        opt.dir_result = os.path.join(dir_root, 'results', opt.dataset, opt.exp_id)

        ####################################################################################################
        """ Model Architecture """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus)
        opt.net = get_model(3, 1, opt.arch)
        opt.param = "%.2fM" % (sum(x.numel() for x in opt.net.parameters()) / 1e+6)
        opt.device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        if opt.gpus is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')
        opt.net.to(device=opt.device)
        ####################################################################################################
        """ Optimizer """
        if opt.optim == "Adam":
            opt.optimizer = optim.Adam(opt.net.parameters(), lr=opt.lr, weight_decay=opt.l2)
        elif opt.optim == "SGD":
            opt.optimizer = optim.SGD(opt.net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.l2)
        ####################################################################################################
        """ Scheduler """
        if opt.sche == "ExpLR":
            gamma = 0.95
            opt.scheduler = torch.optim.lr_scheduler.ExponentialLR(opt.optimizer, gamma=gamma, last_epoch=-1)
        elif opt.sche == "MulStepLR":
            milestones = [90, 120]
            opt.scheduler = torch.optim.lr_scheduler.MultiStepLR(opt.optimizer, milestones=milestones, gamma=0.1)
        elif opt.sche == "CosAnnLR":
            t_max = 5
            opt.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.optimizer, T_max=t_max, eta_min=0.)
        elif opt.sche == "ReduceLR":
            mode = "max"
            factor = 0.9
            patience = 10
            opt.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt.optimizer, mode=mode, factor=factor,
                                                                       patience=patience)
        ####################################################################################################
        """ Loss Function """
        if opt.loss == "dice_loss":
            opt.loss_function = DiceLoss()
        elif opt.loss == "dice_bce_loss":
            opt.loss_function = DiceBCELoss()
        ####################################################################################################

        return opt

    def init(self):
        return self.parse_arg()

