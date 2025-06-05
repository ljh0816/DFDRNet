# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList

"""
    baseline:
    深度监督任务 seg+cls
"""

@MODELS.register_module()
class DFDRHeadV1(BaseDecodeHead):
    """Decode head for DFDRNet.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)

        self.aux_seg_head = self._make_base_head(self.in_channels // 2,self.channels)
        self.aux_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)



        # 消融 stage 4 :(bs, channels * 8, 32, 32) -> (bs, channels * 2, 128, 128)
        # self.aux_multi_cls_head = self._make_base_head(self.in_channels // 2,self.channels)
        # self.aux_multi_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

        # todo
        self.aux_multi_cls_head = self._make_multi_cls_head(self.in_channels*2 , self.channels)
        # self.aux_multi_cls_head = MLPClsHead(self.in_channels *2,self.channels,self.out_channels,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.aux_multi_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            # X_temp_seg, X_temp_mutil_cls, x
            temp_seg_feat,temp_cls_feat,final_feat = inputs

            # input
            #       temp_seg_feat    (bs, channels * 2, 128, 128)
            #       final_feat       (bs, channels * 4, 16, 16)
            #       temp_cls_feat    (bs, channels * 16, 16, 16)

            x_final = self.head(final_feat)
            x_final = self.cls_seg(x_final)

            x_temp_seg = self.aux_seg_head(temp_seg_feat)
            x_temp_seg = self.aux_cls_seg(x_temp_seg)

            x_temp_cls=self.aux_multi_cls_head(temp_cls_feat)
            x_temp_cls=self.aux_multi_cls_seg(x_temp_cls)

            return x_final, x_temp_cls, x_temp_seg

        else:
            x = self.head(inputs)
            x = self.cls_seg(x)

            return x

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:

        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]

        return nn.Sequential(*layers)

    def _make_multi_cls_head(self, in_channels: int,
                             channels: int) -> nn.Sequential:
        #input  (bs, 32 * 16, 16, 16)
        #output (bs, channels, 1, 1)

        layers = [
            nn.AdaptiveAvgPool2d((1, 1)),

            ConvModule(
                in_channels,
                channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]



        return nn.Sequential(*layers)


    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        # x_final, x_temp_cls, x_temp_seg
        final_logit, temp_cls_logit, temp_seg_logit = seg_logits

        seg_label = self._stack_batch_gt(batch_data_samples)

        multi_cls_label=self._stack_batch_multi_cls_gt(seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index)

        final_logit = resize(
            final_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        temp_seg_logit = resize(
            temp_seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # temp_cls_logit = resize(
        #     temp_cls_logit,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        temp_cls_logit=temp_cls_logit.squeeze()

        loss['loss_normal_seg'] = self.loss_decode[0](final_logit, seg_label)
        loss['loss_auxiliary_seg'] = self.loss_decode[1](temp_seg_logit, seg_label)
        loss['loss_auxiliary_cls'] = self.loss_decode[2](temp_cls_logit, multi_cls_label)
        # loss['loss_auxiliary_cls'] = multilabel_categorical_crossentropy(multi_cls_label,temp_cls_logit)
        loss['acc_seg'] = accuracy(final_logit, seg_label, ignore_index=self.ignore_index)

        return loss

    def _stack_batch_multi_cls_gt(self, seg_label, num_classes, ignore_index):
        """
            通过gt分割掩码获得多分类任务标签
        :param seg_label:
        :param num_classes:
        :param ignore_index:
        :return:
        """
        # 初始化一个形状为 [batchsize , num_classes] 的张量
        label_counts = torch.zeros((seg_label.shape[0], num_classes), dtype=torch.int,device=seg_label.device)

        # 遍历每一张掩码图
        for i in range(seg_label.shape[0]):
            # 获取当前掩码图
            mask = seg_label[i]  # 形状为 [1024, 1024]

            # 统计标签，忽略 pad 值 (ignore_index)
            for label in range(num_classes):
                label_counts[i, label] = ((mask == label).sum()!=0)  # 统计每个标签的个数

        return label_counts.float()

