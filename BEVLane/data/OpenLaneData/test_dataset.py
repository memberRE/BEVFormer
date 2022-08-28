import sys

import easydict
import os.path as osp
from Load_Data import *
def _get_train_dataset():
    # args = self.args
    args = easydict.EasyDict()
    base_dir = sys.path[0]
    args.dataset_dir =osp.join(base_dir, 'datasets', 'openlane')
    args.data_dir = osp.join(args.dataset_dir, 'lane3d_300', 'training')

    train_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'training/', args, data_aug=True, save_std=True, seg_bev=True)

    # train_dataset.normalize_lane_label()
    train_loader, train_sampler = get_loader(train_dataset, args)

    return train_dataset, train_loader, train_sampler

def _get_valid_dataset():
    args = easydict.EasyDict()
    base_dir = sys.path[0]
    args.dataset_dir = osp.join(base_dir, 'datasets', 'openlane')
    args.data_dir = osp.join(args.dataset_dir, 'lane3d_300', 'validation')
    valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'validation/', args, seg_bev=True)

    # assign std of valid dataset to be consistent with train dataset
    valid_dataset.set_x_off_std(self.train_dataset._x_off_std)
    if not args.no_3d:
        valid_dataset.set_z_std(self.train_dataset._z_std)
    # valid_dataset.normalize_lane_label()
    valid_loader, valid_sampler = get_loader(valid_dataset, args)

    return valid_dataset, valid_loader, valid_sampler