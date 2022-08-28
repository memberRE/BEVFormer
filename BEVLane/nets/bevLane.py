import copy
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
# from mmdet3d.core.bbox.coders import build_bbox_coder
# from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models import builder
from mmcv.runner import BaseModule
import torch.distributed as dist
# from projects.mmdet3d_plugin.models.utils import GridMask
import BEVLane.nets


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    @auto_fp16()
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)

@DETECTORS.register_module()
class PersFormerOneStage(BaseModule):

    def __init__(self, use_grid_mask=True, img_backbone=None, img_neck=None, pts_bbox_head=None, img_roi_head=None,
                 img_rpn_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None    # TODO:Why to update cfg.pts
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_img_feat(self, img, img_metas = None, len_queue=None):
        """Extract features of images."""
        # img.shape = [BCHW]
        # return list(BNCHW or B len N CHW), note: len(list(...)) features,
        B = img.size(0) # BCHW or B1CHW
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, C, H, W)  # BS和len本质上是一回事
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            # img_feats_list = list( B lenQ 1CHW) len(list) = 4, original is list(B len numC C H W)
            # print(img_metas_list)
            for i in range(len_queue):
                #img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # [B,1,CHW] ([B,numC,CHW])
                prev_bev = self.pts_bbox_head(
                    img_feats, None, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward(self,
                # points=None,
                img_metas=None,
                # gt_bboxes_3d=None,
                # gt_labels_3d=None,
                gt_labels=None, # [BCHW]
                gt_bboxes=None,
                img=None,
                proposals=None,
                gt_bboxes_ignore=None,
                img_depth=None,
                img_mask=None
                ):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...] # [B,N,C,H,W] note: no num_camera
        img = img[:, -1, ...]   # [B,CHW]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        print('prebev_shape:', prev_bev.shape)
        img_metas = None#[each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
        # list(B1CHW)
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev)
        # loss_inputs = [gt_bboxes, gt_labels, outs]  # original is 3D bboxes
        loss_inputs = [gt_labels, outs]
        losses = dict()
        losses_new = self.pts_bbox_head.loss_seg(*loss_inputs, img_metas=img_metas)
        losses.update(losses_new)
        return losses

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        frames, masks, obj_num, info = data
        losses = self(img = frames, gt_labels = masks, img_metas = info)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(info))

        return outputs