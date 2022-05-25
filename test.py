import os
import torch
import argparse
import datetime
import numpy as np
import random
import shutil

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.builders import build_dataset, build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate
from utils.vis_utils import visualize_numpy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

args, cfg = parse_config()
exp_dir = ('/').join(args.ckpt.split('/')[:-2])

random_seed = cfg.RANDOM_SEED # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Build Dataloader
val_dataset = build_dataset(cfg, split='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)
state_dict = torch.load(args.ckpt)
epoch = state_dict['epoch']
net.load_state_dict(state_dict['model_state_dict'])
net = net.cuda()
net.eval()

print('Evaluating Epoch: ', epoch)
val_dict = validate(net, val_dataloader, net.get_loss, 'cuda', is_segmentation = cfg.DATASET.IS_SEGMENTATION)

if cfg.DATASET.IS_SEGMENTATION:
    miou = np.round(val_dict['miou'], 4)
    print('miou', miou)

else:
    val_loss    = np.round(val_dict['loss'], 4)
    val_acc     = np.round(val_dict['acc'], 2)
    val_acc_avg = np.round(val_dict['acc_avg'], 2)

    print('val_loss', val_loss)
    print('val_acc', val_acc)
    print('val_acc_avg', val_acc_avg)


if cfg.DATASET.IS_SEGMENTATION:
    with open(exp_dir + '/eval_best.txt', 'w') as f:
        f.write('Best Epoch: ' + str(epoch))
        f.write('\nBest miou: ' + str(miou))

else:
    with open(exp_dir + '/eval_best.txt', 'w') as f:
        f.write('Best Epoch: ' + str(epoch))
        f.write('\nBest Acc: ' + str(val_acc))
        f.write('\nBest Mean Acc: ' + str(val_acc_avg))
        f.write('\nBest Loss: ' + str(val_loss))


torch.save(state_dict['model_state_dict'], exp_dir + '/ckpt_model_only.pth')

