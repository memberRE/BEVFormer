import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv import ConfigDict, deprecated_api_warning
from mmcv.cnn import xavier_init, build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE, ATTENTION
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_attention, build_feedforward_network, \
    TransformerLayerSequence
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn import ModuleList
from torch.nn.init import normal_
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
# from .spatial_cross_attention import MSDeformableAttention3D
# from .decoder import CustomMSDeformableAttention

@TRANSFORMER_LAYER_SEQUENCE.register_module()   # 假装这是一个TRM
class MMASegmentDecoder(BaseModule):
    def __init__(self,*args, **kwargs):
        super().__init__()

    def forward(self,
                query,
                value,
                *args,
                reference_points=None,
                cls_branches=None,
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            value (Tensor): bs, bev_h*bev_w, embed_dims
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        print('decoder value shape:', value.shape)
        output = value.permute(1, 0, 2) # b , hw, c
        x = cls_branches[0](output)

        return x, None
