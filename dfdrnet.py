# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from torch import Tensor

@MODELS.register_module()
class DFDRNet(BaseModule):
    """
    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels: (int): The base channels of DDRNet. Default: 32.
        ppm_channels (int): The channels of PPM module. Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        # bilateral fusion
        self.ff_1_s = BGAFF(inp=channels * 2,oup=channels * 2,initial_fusion_type='add')
        # self.ff_1_c = BGAFF(inp=channels * 4, oup=channels * 4)
        self.ff_2_s = BGAFF(inp=channels * 2, oup=channels * 2,initial_fusion_type='add')
        # self.ff_2_c = BGAFF(inp=channels * 8, oup=channels * 8)
        self.ff_3 = BGAFF(inp=channels * 4, oup=channels * 4,initial_fusion_type='add')

        # self.ff_1_s = CCAFF(inp=channels * 2, oup=channels * 2)
        # self.ff_1_c = CCAFF(inp=channels * 4, oup=channels * 4)
        # self.ff_2_s = CCAFF(inp=channels * 2, oup=channels * 2)
        # self.ff_2_c = CCAFF(inp=channels * 8, oup=channels * 8)
        # self.ff_3 = CCAFF(inp=channels * 4, oup=channels * 4)

        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        # high resolution(spatial) branch
        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1,
                ))

        self.spp = DAPPM(channels * 16, ppm_channels, channels * 4, num_scales=5)
        # self.spp = DAPPMwithAFF(channels * 16, ppm_channels, channels * 4, num_scales=5)


    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        out_size = (x.shape[-2] // 8, x.shape[-1] // 8)

        # stage 0-2
        x = self.stem(x)

        # stage3
        x_c = self.context_branch_layers[0](x)
        x_s = self.spatial_branch_layers[0](x)
        comp_c = self.compression_1(self.relu(x_c))
        x_c += self.down_1(self.relu(x_s))
        # x_c = self.ff_1_c(x_c, self.down_1(self.relu(x_s)))
        x_c_upsampled = resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_s=self.ff_1_s(x_s,x_c_upsampled)
        # x_s = x_s+x_c_upsampled

        if self.training:
            # temp_context = x_s.clone()
            X_temp_seg= x_s.clone()

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c)) #(6, channels * 8, 32, 32)
        x_s = self.spatial_branch_layers[1](self.relu(x_s)) #(6, channels * 2, 128, 128)
        comp_c = self.compression_2(self.relu(x_c))
        x_c += self.down_2(self.relu(x_s))
        # x_c = self.ff_2_c(x_c,self.down_2(self.relu(x_s)))
        x_c_upsampled= resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_s = self.ff_2_s(x_s, x_c_upsampled)
        # x_s = x_s + x_c_upsampled

        if self.training:
            X_temp_multi_cls = x_c.clone()

        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s))#(6, channels * 4, 128, 128)
        x_c = self.context_branch_layers[2](self.relu(x_c))#(6, channels * 16, 16, 16)
        x_c = self.spp(x_c) #(6, channels * 4, 16, 16)
        x_c = resize(
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_f=self.ff_3(x_s, x_c)
        # x_f = x_s+x_c

        return (X_temp_seg, X_temp_multi_cls, x_f) if self.training else x_f
        # return (temp_context, x_f) if self.training else x_f


#坐标注意力
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class BGAFF(nn.Module):
    """
         local attention, global attention,
    """
    def __init__(self,
                 inp,
                 oup,
                 reduction=32,
                 initial_fusion_type='add',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__()
        assert initial_fusion_type in ['add', 'cat']
        self.initial_fusion_type=initial_fusion_type
        mip = max(8, inp // reduction)

        if self.initial_fusion_type=="cat":
            self.conv_concat=ConvModule(
                                inp*2,
                                inp,
                                kernel_size=1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        else:
            self.conv_concat =None


        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(oup),
        )


        # global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(oup),
        )
        self.sigmoid = nn.Sigmoid()

        # coordinate attention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):

        if self.initial_fusion_type=="add":
            z = x + y
        else:
            z = torch.cat([x, y], dim=1)
            z = self.conv_concat(z)

        n, c, h, w = z.size()

        xl = self.local_att(z)
        xg = self.global_att(z)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        z_h = self.pool_h(z)
        z_w = self.pool_w(z).permute(0, 1, 3, 2)

        z = torch.cat([z_h, z_w], dim=2)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.act(z)

        z_h, z_w = torch.split(z, [h, w], dim=2)
        z_w = z_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(z_h).sigmoid()
        a_w = self.conv_w(z_w).sigmoid()
        atten = a_w * a_h + wei

        out = x * atten + y * (1- atten)
        return out









