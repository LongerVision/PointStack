#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import json
import h5py
from glob import glob
import numpy as np

from .dataset_template import DatasetTemplate

class PartNormal(DatasetTemplate):
    def __init__(self, cfg, class_choice=None, split='train', load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False):
        super().__init__(cfg = cfg, class_choice=None, split=split, load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False)
        self.npoints = cfg.DATASET.NUM_POINTS
        self.root = self.data_dir
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = False

        self.split = split

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        # if self.normalize:
        #     point_set = pc_normalize(point_set)

        choice = np.random.choice(min(len(seg), self.npoints), self.npoints, replace=True)
        #choice = np.linspace(0, self.num_points, num=self.num_points).astype(int)
        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        data_dic = {
            'points'    : torch.from_numpy(point_set).cuda(),
            'seg_id'    : torch.from_numpy(seg).cuda(),
            'cls_tokens': torch.from_numpy(cls).cuda(),
            'norms'     : torch.from_numpy(normal).cuda()
        }

        return data_dic