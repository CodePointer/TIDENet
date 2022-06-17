# -*- coding: utf-8 -*-

# @Time:      2021/8/12 14:52
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      ride_net.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import torch.nn as nn
import torch.nn.functional as F

from tide.submodules import SmallUpdateBlock, SmallEncoder
from tide.submodules import CorrBlock1D, coords_grid, upflow8, dnflow8


# - Coding Part - #
class RIDEInit(nn.Module):
    def __init__(self, idim=2, fdim=128, hdim=96, cdim=64,
                 temp_type='gru', mask_flag=False, last_pred=False, iter_times=1):
        super(RIDEInit, self).__init__()
        self.ride_feature = RIDEFeature(idim, fdim)
        self.ride_update = RIDEUpdate(idim, hdim, cdim, temp_type, mask_flag, last_pred, iter_times)

    def forward(self, img, pat):
        fmap_pat = self.ride_feature(img=pat)
        fmap_img = self.ride_feature(img=img)

        disps, _, _ = self.ride_update(fmap_img, fmap_pat, img)
        return disps[0]


class RIDEFeature(nn.Module):
    def __init__(self, idim=2, fdim=128):
        super(RIDEFeature, self).__init__()

        self.fnet = SmallEncoder(input_dim=idim, output_dim=fdim, norm_fn='instance')

    def forward(self, img):
        return self.fnet(img)


class RIDEHidden(nn.Module):
    def __init__(self, idim=2, hdim=96):
        super(RIDEHidden, self).__init__()
        self.hnet = SmallEncoder(input_dim=idim, output_dim=hdim, norm_fn='none')

    def forward(self, img):
        return torch.tanh(self.hnet(img))


class RIDEUpdate(nn.Module):
    def __init__(self, idim=2, hdim=96, cdim=64, temp_type='gru', mask_flag=False, last_pred=False, iter_times=1):
        super(RIDEUpdate, self).__init__()

        self.hidden_dim = hdim
        self.context_dim = cdim
        self.corr_levels = 4
        self.corr_radius = 3
        self.iter = iter_times
        self.temp_type = temp_type

        self.cnet = SmallEncoder(input_dim=idim, output_dim=cdim, norm_fn='none')
        self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius,
                                             temp=self.temp_type, hidden_dim=hdim, mask_flag=mask_flag)
        self.last_pred = last_pred

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_but_mask(self):
        self.update_block.freeze_but_mask()

    @staticmethod
    def initialize_flow(img):
        """ Flow is represented as difference between two coordinate flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coord0 = coords_grid(N, H, W).to(img.device)
        coord1 = coords_grid(N, H, W).to(img.device)
        # pattern flow computed as difference: flow = coords1 - coords0
        return coord0, coord1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        if mask is None:
            return upflow8(flow)
        else:
            N, _, H, W = flow.shape
            rad = 3
            mask = mask.view(N, 1, rad * rad, 8, 8, H, W)
            mask = torch.softmax(mask, dim=2)

            up_flow = nn.functional.unfold(8 * flow, [rad, rad], padding=1)
            up_flow = up_flow.view(N, 1, rad * rad, 1, 1, H, W)

            up_flow = torch.sum(mask * up_flow, dim=2, keepdim=False)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def forward(self, fmap1, fmap2, img1, flow_init=None, h=None, c=None):

        corr_fn = CorrBlock1D(fmap1, fmap2, radius=self.corr_radius)

        # Context network
        inp = torch.relu(self.cnet(img1))
        if h is None:
            h = torch.zeros(fmap1.shape[0], self.hidden_dim, fmap1.shape[2], fmap1.shape[3]).to(fmap1.device)
        if c is None and self.temp_type == 'lstm':
            c = torch.zeros(fmap1.shape[0], self.hidden_dim, fmap1.shape[2], fmap1.shape[3]).to(fmap1.device)

        coords0, coords1 = self.initialize_flow(fmap1)

        if flow_init is not None:
            if flow_init.shape[2] == coords0.shape[2]:
                coords1 = coords0 - flow_init
            else:
                coords1 = coords0 - dnflow8(flow_init)

        flow_predictions = []
        for itr in range(self.iter):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords0 - coords1
            h, c, mask, delta_flow = self.update_block(inp, corr, flow, net_h=h, net_c=c)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if self.last_pred:
                continue

            # upsample predictions
            flow_up = self.upsample_flow(coords0 - coords1, mask)

            flow_predictions.append(flow_up)

        if self.last_pred:
            return self.upsample_flow(coords0 - coords1, mask), h, c
        else:
            return flow_predictions, h, c
