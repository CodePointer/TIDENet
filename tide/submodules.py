# -*- coding: utf-8 -*-

# @Time:      2021/3/10 16:32
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      submodules.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import torch.nn as nn
import torch.nn.functional as F


# - Coding Part - #
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, input_dim=192 + 128, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvLSTM, self).__init__()
        self.convi = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convf = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convo = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convg = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h, c):
        hx = torch.cat([h, x], dim=1)

        i = torch.sigmoid(self.convi(hx))
        f = torch.sigmoid(self.convf(hx))
        o = torch.sigmoid(self.convo(hx))

        cc_g = torch.layer_norm(self.convg(hx), h.shape[-2:])
        g = torch.celu(cc_g)

        c_next = f * c + i * g
        c_next = torch.layer_norm(c_next, h.shape[-2:])
        h_next = o * torch.celu(c_next)

        return h_next, c_next


class SmallMotionEncoder(nn.Module):
    def __init__(self, corr_levels=0, corr_radius=0, idim=None):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        if idim is not None:
            cor_planes = idim
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, temp='gru', hidden_dim=96, mask_flag=False, mask_rad=3):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(corr_levels, corr_radius)
        self.temporal_type = temp
        input_dim = 80 + 1 + 64
        if temp == 'gru':
            self.temp = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        elif temp == 'lstm':
            self.temp = ConvLSTM(hidden_dim=hidden_dim, input_dim=input_dim)
        else:
            raise NotImplementedError(f'Invalid temp={temp}')
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

        self.mask = None
        if mask_flag:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 8 * 8 * mask_rad * mask_rad, 1, padding=0)
            )

    def freeze_but_mask(self):
        for param in self.parameters():
            param.requires_grad = False
        # for param in self.flow_head.parameters():
        #     param.requires_grad = True
        if self.mask is not None:
            for param in self.mask.parameters():
                param.requires_grad = True

    def forward(self, inp, corr, flow, net_h, net_c=None):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        if self.temporal_type == 'gru':
            net_h = self.temp(x=inp, h=net_h)
        elif self.temporal_type == 'lstm':
            net_h, net_c = self.temp(x=inp, h=net_h, c=net_c)

        delta_flow = self.flow_head(net_h)

        mask = None
        if self.mask is not None:
            mask = self.mask(net_h)

        return net_h, net_c, mask, delta_flow


class SmallResidualBlock(nn.Module):
    def __init__(self, fdim=128, hdim=96):
        super(SmallResidualBlock, self).__init__()
        self.hidden_dim = hdim
        self.encoder = SmallMotionEncoder(idim=fdim)
        self.gru = ConvGRU(hidden_dim=hdim, input_dim=81)
        self.flow_head = FlowHead(hdim, hidden_dim=128)
        pass

    def create_hidden(self, inp):
        return torch.zeros(inp.shape[0], self.hidden_dim, inp.shape[2], inp.shape[3]).to(inp.device)

    def forward(self, fmap, disp, hidden=None):
        hidden = hidden if hidden is not None else self.create_hidden(fmap)

        disp_features = self.encoder(disp, fmap)
        hidden = self.gru(hidden, disp_features)
        delta_disp = self.flow_head(hidden)

        return hidden, delta_disp


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class SmallEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=128, norm_fn='batch'):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        # output_dim = 128
        # conv1_out = 32
        # layer1_out = 32
        # layer2_out = 64
        # layer3_out = 96

        output_dim = 256
        conv1_out = 64
        layer1_out = 64
        layer2_out = 128
        layer3_out = 256

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(conv1_out)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(conv1_out)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, conv1_out, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = conv1_out
        self.layer1 = self._make_layer(layer1_out, stride=1)
        self.layer2 = self._make_layer(layer2_out, stride=2)
        self.layer3 = self._make_layer(layer3_out, stride=2)

        self.conv2 = nn.Conv2d(layer3_out, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        x = self.conv1(x)   # input_dim -> 32
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 32 -> 32
        x = self.layer2(x)  # 32 -> 64
        x = self.layer3(x)  # 64 -> 96
        x = self.conv2(x)   # 96 -> 128

        return x


class LargeEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=128, norm_fn='batch'):
        super(LargeEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=11, stride=2, padding=5)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)

        self.conv2 = nn.Conv2d(256, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        return x


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool1d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        :param coords: [N, 1, H, W]
        :return: [N, num_levels, H, W]
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, hei, wid, _ = coords.shape   # [N, H, W, 1]

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1).reshape(1, -1).repeat(2 * r + 1, 1).to(coords.device)

            centroid_lvl = coords.reshape(batch * hei * wid, 1, 1, 1) / 2 ** i
            delta_lvl = dx.view(1, 2 * r + 1, 2 * r + 1, 1)
            coords_lvl = centroid_lvl + delta_lvl           # [N * H * W, 2r+1, 2r+1, 1]

            corr = bilinear_sampler1d(corr, coords_lvl)
            corr = corr.view(batch, hei, wid, -1)   # [N, H, W, (2r+1) ** 2]
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)    # [N, H, W, num_levels * (2r+1) ** 2]
        return out.permute(0, 3, 1, 2).contiguous().float()  # [N, num_levels, H, W]

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, hei, wid = fmap1.shape
        fmap1 = fmap1.transpose(1, 2).reshape(batch * hei, dim, wid)
        fmap2 = fmap2.transpose(1, 2).reshape(batch * hei, dim, wid)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.reshape(batch, hei, wid, 1, wid)

        return corr / torch.sqrt(torch.tensor(dim).float() + 1e-8)


def bilinear_sampler1d(corr, coord, mode='bilinear'):   # 问题：Grid需要进行sample https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    """
    :param corr: [N*H*W, 1, wid]
    :param coord: [N*H*W, 2r+1, 2r+1, 1]
    :param mode:
    :return:
    """
    wid = corr.shape[2]
    img = corr.reshape(corr.shape[0], corr.shape[1], 1, corr.shape[2])      # [NHW, 1, 1, wid]
    grid = torch.cat([2.0 * (coord / (wid - 1)) - 1.0, torch.zeros_like(coord)], dim=3)  # [NHW, 2r+1, 2r+1, 2]
    img = F.grid_sample(img, grid, align_corners=False)     # [NHW, 1, 2r+1, 2r+1]
    return img


def coords_grid(batch, hei, wid):
    xx_grid = torch.arange(0, wid).reshape(1, -1).repeat(hei, 1)
    return xx_grid.reshape(1, 1, hei, wid).repeat(batch, 1, 1, 1).float()


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=False)


def dnflow8(flow, mode='nearest'):
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode) / 8.0
