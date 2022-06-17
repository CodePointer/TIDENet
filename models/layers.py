# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import torch.nn.functional as F


# - Coding Part - #
class LCN(torch.nn.Module):
    """Local Contract Normalization"""

    def __init__(self, radius, epsilon):
        super().__init__()

        self.epsilon = epsilon
        self.radius = radius
        self.pix_num = (2 * self.radius + 1) ** 2

        self.avg_conv = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radius),
            torch.nn.Conv2d(1, 1, kernel_size=2 * radius + 1, bias=False)
        )
        self.avg_conv[1].weight.requires_grad = False
        self.avg_conv[1].weight.fill_(1. / self.pix_num)

    def forward(self, data):
        self.avg_conv.to(data.device)
        avg = self.avg_conv(data)

        diff = data - avg
        std2 = self.avg_conv(diff ** 2)
        std = torch.sqrt(std2)

        img = (data - avg) / (std + self.epsilon)
        return img, std


class WarpLayer(torch.nn.Module):
    """Warp mat to another according to disp"""
    def __init__(self, height, width, device=None):
        super().__init__()
        self.H = height
        self.W = width

        # Create xy_grid
        xx = torch.arange(0, self.W).view(1, -1).repeat(self.H, 1)
        yy = torch.arange(0, self.H).view(-1, 1).repeat(1, self.W)
        xx = xx.view(1, self.H, self.W, 1)
        yy = yy.view(1, self.H, self.W, 1)
        self.xy_grid = torch.cat((xx, yy), 3).float()
        if device is not None:
            self.xy_grid = self.xy_grid.to(device)

    def forward(self, disp_mat, src_mat, mask_flag=False):
        """Warp src_mat according to disp_mat"""
        self.xy_grid = self.xy_grid.to(disp_mat.device)
        xy_grid = self.xy_grid.expand(disp_mat.shape[0], *self.xy_grid.shape[1:])
        disp1d = disp_mat.permute(0, 2, 3, 1)
        disp2d = torch.cat((disp1d, torch.zeros_like(disp1d)), dim=3)

        xy_grid = xy_grid - disp2d
        xy_grid[:, :, :, 0] = 2.0 * (xy_grid[:, :, :, 0] / (self.W - 1) - 0.5)
        xy_grid[:, :, :, 1] = 2.0 * (xy_grid[:, :, :, 1] / (self.H - 1) - 0.5)

        warped_mat = torch.nn.functional.grid_sample(src_mat, xy_grid, padding_mode='border', align_corners=False)
        if mask_flag:
            mask = torch.nn.functional.grid_sample(torch.ones_like(src_mat), xy_grid, padding_mode='border',
                                                   align_corners=False)
            mask = torch.floor(torch.clamp(mask, 0.0, 1.0))
            warped_mat *= mask

        return warped_mat
