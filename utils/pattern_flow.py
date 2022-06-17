# -*- coding: utf-8 -*-

# @Time:      2022/6/17 19:02
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      pattern_flow.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from utils.pointerlib import t2a, a2t


# - Coding Part - #
class PFlowEstimatorLK:
    def __init__(self, hei, wid, win_rad=15, flag_tensor=True):
        self.lk_params = {'winSize': (win_rad, win_rad),
                          'maxLevel': 0,
                          'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        self.imsize = (hei, wid)
        self.flag_tensor = flag_tensor

        # Set old points
        xx = np.arange(0, wid).reshape(1, -1).repeat(hei, 0).astype(np.float32)
        yy = np.arange(0, hei).reshape(-1, 1).repeat(wid, 1).astype(np.float32)
        xy = np.stack([xx, yy], axis=2)
        self.cp_mat = cv2.resize(xy, (wid // 8, hei // 8), interpolation=cv2.INTER_LINEAR)
        self.p0 = self.cp_mat.reshape(-1, 2)

        self.thd_set = {'y': 1.0, 'bk': 0.5}

    def set_thd(self, **kwargs):
        for key in kwargs:
            self.thd_set[key] = kwargs[key]

    def _estimate_dn8_np(self, src_img, dst_img):
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(src_img, dst_img, self.p0, None, **self.lk_params)
        p0, st0, err0 = cv2.calcOpticalFlowPyrLK(dst_img, src_img, p1, None, **self.lk_params)

        # 0. Filter by st0
        st1[st0 == 0] = 0

        # 1. Filter by delta_y
        st1[np.abs(p1[:, 1] - self.p0[:, 1]) > self.thd_set['y']] = 0
        st1[np.abs(p1[:, 1] - p0[:, 1]) > self.thd_set['y']] = 0

        # 2. Filter by p0 & self.p0
        st1[np.abs(p0[:, 0] - self.p0[:, 0]) > self.thd_set['bk']] = 0

        # Apply to x
        delta = p1 - self.p0
        delta_xx = delta[:, :1]
        delta_xx[st1 == 0] = 0.0
        return delta_xx, st1, err1

    def estimate_dn8(self, src_img, dst_img):
        if self.flag_tensor:
            batch, ch, hei, wid = src_img.shape
        else:
            batch, ch = 1, 1

        pf_mats = []
        st_mats = []
        for n in range(batch):
            if self.flag_tensor:
                delta_xx, st, err = self._estimate_dn8_np(t2a(src_img[n] * 255.0).astype(np.uint8),
                                                          t2a(dst_img[n] * 255.0).astype(np.uint8))
            else:
                delta_xx, st, err = self._estimate_dn8_np(src_img, dst_img)
            pf_mat = delta_xx.reshape(self.cp_mat.shape[:2])
            st_mat = st.reshape(self.cp_mat.shape[:2])
            pf_mats.append(a2t(pf_mat))
            st_mats.append(a2t(st_mat))

        if self.flag_tensor:
            return torch.stack(pf_mats, dim=0).to(src_img.device), torch.stack(st_mats, dim=0).to(src_img.device)
        else:
            return pf_mats[0], st_mats[0]


class ConIdGenerator:
    def __init__(self, start_id=42):
        self._start_id = start_id
        self.next_id = start_id + 1

    def reset(self):
        self.next_id = self._start_id + 1

    def get_new(self, length=1):
        next_ids = list(range(self.next_id, self.next_id + length))
        self.next_id += length
        if length == 1:
            return next_ids[0]
        else:
            return next_ids

    def get_new_tensor(self, length=1):
        res = self.get_new(length)
        if length == 1:
            res = [res]
        return torch.Tensor(res).long()


class MPFlowEstimator:
    """
    Flow type:
    [x^t-T+1, ..., x^t-1, x^t, idx, y, L]
    """

    def __init__(self, temp_win, sig2_max=1024.0, device='cpu'):
        self.T = temp_win
        self.sig2_max = sig2_max
        self.id_generator = ConIdGenerator()
        self.w_rad = 7
        self.h_rad = 1
        self.device = device
        kernel = torch.zeros(1, 1, 2 * self.h_rad + 1, 2 * self.w_rad + 1)
        for dh in range(-self.h_rad, self.h_rad + 1, 1):
            for dw in range(-self.w_rad, self.w_rad + 1, 1):
                h, w = dh + self.h_rad, dw + self.w_rad
                dist = dh ** 2 + dw ** 2
                norm_alpha = (self.h_rad ** 2 + self.w_rad ** 2) / 10
                kernel[0, 0, h, w] = np.exp(-dist / norm_alpha)
        self.kernel = kernel.to(device)
        self.imsize = None

    def reset(self):
        self.id_generator.reset()

    def _find_nearest_point(self, mask_center, idx_map_lst):
        hei, wid = self.imsize

        def get_nearest(src_idx_map):
            src_idx_map = src_idx_map.unsqueeze(0).unsqueeze(1).float()

            src_one_map = src_idx_map.clone()
            src_one_map[src_one_map > 0] = 1.0

            one_unfold = F.unfold(src_one_map, kernel_size=self.kernel.shape[2:4], padding=(self.h_rad, self.w_rad))
            one_unfold = one_unfold.reshape(1, -1, hei, wid)
            kernel_1d = self.kernel.reshape(1, -1, 1, 1)
            _, ref_pos_mat = torch.max(one_unfold * kernel_1d, dim=1, keepdim=True)

            idx_unfold = F.unfold(src_idx_map, kernel_size=self.kernel.shape[2:4], padding=(self.h_rad, self.w_rad))
            idx_unfold = idx_unfold.reshape(1, -1, hei, wid)
            idx_nearest = torch.gather(idx_unfold, dim=1, index=ref_pos_mat).squeeze()
            return idx_nearest

        idx_nearest_now = get_nearest(idx_map_lst)
        idx_nearest_now[mask_center == 0] = 0

        # Filter out unique
        _, x_coord, y_coord = self._map2dot(mask_center)
        id_now = torch.arange(0, x_coord.shape[0]).to(self.device)
        idx_map_now = self._dot2map(torch.stack([id_now, x_coord, y_coord], dim=1))

        nearest_now2lst = get_nearest(idx_map_now)
        nearest_now2lst[idx_map_lst == 0] = 0
        nearest_back = get_nearest(nearest_now2lst)
        mask_invalid = (nearest_back != idx_map_now)

        idx_nearest_now[mask_invalid] = 0
        return idx_nearest_now

    def _map2dot(self, idx_map):
        idx_map = idx_map.squeeze()
        y_coord, x_coord = torch.where(idx_map > 0)
        value = idx_map[y_coord.long(), x_coord.long()]
        return value, x_coord, y_coord

    def _dot2map(self, dot_coord):
        if dot_coord.shape[1] == 3:
            id_set, x_coord, y_coord = torch.chunk(dot_coord, chunks=3, dim=1)
        elif dot_coord.shape[1] == 2:
            x_coord, y_coord = torch.chunk(dot_coord, chunks=2, dim=1)
            id_set = torch.ones_like(x_coord)
        else:
            return
        dot_mat = torch.zeros(self.imsize).to(self.device)
        dot_mat = dot_mat.index_put(indices=[y_coord.long(), x_coord.long()], values=id_set.float())
        return dot_mat

    def run(self, mask_centers, distribution=None):
        """
        mask_centers: [T, 1, H, W]
        distribution: [Kc', 5]; idx, x^t-T, y^t-T, mu, sig2

        Initialization:
            mu: <random>
            sig2: sig2_max
        """
        self.imsize = mask_centers[0].squeeze().shape

        res_list = []
        for i in range(self.T):
            mask_centers[i] = mask_centers[i].to(self.device)
            _, x_coord, y_coord = self._map2dot(mask_centers[i])

            if distribution is None and i == 0:
                idx_set = self.id_generator.get_new_tensor(x_coord.shape[0]).float()
                idx_set = idx_set.to(self.device)

            else:
                last_dot_set = res_list[-1] if i > 0 else distribution[:, :3]  # [Kc', 3]; idx, x^t-T, y^t-T
                valid_dot, = torch.where(last_dot_set[:, 1] >= 0)
                idx_mat_lst = self._dot2map(last_dot_set[valid_dot, :])
                idx_mat = self._find_nearest_point(mask_centers[i].squeeze(), idx_mat_lst)
                idx_set = idx_mat[y_coord, x_coord]
                new_pos, = torch.where(idx_set <= 0)
                idx_set[new_pos] = self.id_generator.get_new_tensor(new_pos.shape[0]).float().to(self.device)

            dot_set = torch.stack([idx_set, x_coord, y_coord], dim=1)  # [Kc', 3]
            res_list.append(dot_set)

        min_val = int(res_list[0][:, 0].min().item())
        max_val = max([int(res_list[t][:, 0].max().item()) for t in range(self.T)]) + 1
        flow_set = -torch.ones([self.T, 2, max_val - min_val + 1]).to(self.device)  # [T, 2, Kc]; 0 -> invalid
        for t in range(self.T):
            idx = res_list[t][:, 0].long() - min_val + 1
            flow_set[t, 0, idx] = res_list[t][:, 1]
            flow_set[t, 1, idx] = res_list[t][:, 2]

        new_distribution = torch.zeros([max_val - min_val + 1, 5]).to(self.device)  # [Kc, 3]
        mask_match = torch.zeros(max_val - min_val + 1).to(self.device)
        new_distribution[:, 0] = torch.arange(min_val - 1, max_val)  # idx
        new_distribution[:, 1] = flow_set[-1, 0, :]  # x^t
        new_distribution[:, 2] = flow_set[-1, 1, :]  # y^t
        if distribution is None:
            mask_match[:] = 0.0
        else:
            idx_lst = distribution[:, 0].long() - min_val + 1
            idx_lst[idx_lst < 0] = 0
            mask_match[idx_lst] = 1.0
            new_distribution[idx_lst, 3] = distribution[:, 3]
            new_distribution[idx_lst, 4] = distribution[:, 4]
        mask_match[0] = 0.0
        mask_match = mask_match.view(-1, 1)  # [Kc, 1]

        # Init new
        mu_init = torch.ones_like(mask_match).to(self.device) * self.imsize[1] / 2.0
        sig2_init = torch.ones_like(mask_match).to(self.device) * self.sig2_max
        init_set = torch.cat([mu_init, sig2_init], dim=1)  # [Kc, 2]
        new_distribution[:, 3:5] *= mask_match
        new_distribution[:, 3:5] += (1.0 - mask_match) * init_set
        new_distribution[0, :] = torch.Tensor([0.0, -1.0, -1.0, self.imsize[1] / 2.0, self.sig2_max]).to(self.device)

        return flow_set, new_distribution
