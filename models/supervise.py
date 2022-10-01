# -*- coding: utf-8 -*-

# @Time:      2021/03/03 21:23
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      supervise.py
# @Software:  VSCode
# @Description:
#   Losses for training.

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np
from models.layers import WarpLayer


# - Coding Part - #
class BaseLoss(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class SuperviseDistLoss(BaseLoss):
    """L1 or L2 loss for supervision"""
    def __init__(self, name='DEFAULT', dist='l1'):
        super().__init__(name)
        self.dist = dist
        self.crit = None
        if dist == 'l1':
            self.crit = torch.nn.L1Loss(reduction='none')
        elif dist == 'l2':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif dist == 'smoothl1':
            self.crit = torch.nn.SmoothL1Loss(reduction='none')
        elif dist == 'bce':
            self.crit = torch.nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown loss type: {dist}')

    def forward(self, pred, target, mask=None):
        """
        disp_prd: [N, 1, H, W]
        """

        err_map = self.crit(pred, target)
        if mask is None:
            mask = torch.ones_like(pred)
        val = (err_map * mask).sum() / (mask.sum() + 1e-8)
        return val, err_map


class WarpedPhotoLoss(BaseLoss):
    def __init__(self, hei, wid, name='DEFAULT', dist='l1'):
        super().__init__(name)
        self.hei, self.wid = hei, wid
        self.warp = WarpLayer(self.hei, self.wid)
        self.dist = dist

        self.crit = None
        if dist == 'l1':
            self.crit = torch.nn.L1Loss(reduction='none')
        elif dist == 'l2':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif dist == 'smoothl1':
            self.crit = torch.nn.SmoothL1Loss(reduction='none')
        elif dist == 'bce':
            self.crit = torch.nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown loss type: {dist}')

    def forward(self, img_dst, img_src, disp_mat, mask=None, std=None):
        if mask is None:
            mask = torch.ones_like(img_dst)

        pat_lcn = img_src
        img_lcn = img_dst
        img_wrp = self.warp(disp_mat=disp_mat, src_mat=pat_lcn)
        img_err = self.crit(img_wrp, img_lcn.view(img_wrp.shape))

        mask_for_err = mask
        val = (img_err * mask_for_err).sum() / mask_for_err.sum()
        return val, img_wrp, img_err, img_lcn, mask_for_err


class PFDistLoss(BaseLoss):

    def __init__(self, pat_info, clip_len, hei, wid, name='DEFAULT', dist='l1', device=None):
        super().__init__(name)

        # self.pat_xp = pat_xp  # [1, 1, H, W], float
        # self.pat_pid = pat_pid  # [1, 1, H, W], int
        # self.pat_blob = pat_blob
        # for x in pat_blob.keys():
        #     self.pat_blob[x] = self.pat_blob[x].to(device)
        self.pat_info = pat_info
        self.pat_distribution = None

        # self.thred_max_diff = 3.0
        self.T = clip_len
        self.num_pat = self.pat_info['pos'].shape[2]  # [1, 1, Kp, 2]
        self.H = hei
        self.W = wid
        self.device = device
        self.args = {
            'sig2_prior_add': 10.0,
            "max_sig2": 1024.0,
            "exp_sig2": 1.0,
            'edge_thr': 15.0,
            'alpha_edge': 1.0,
        }

        def gaussian_kernel(rad=2, sig=1.):
            length = 2 * rad + 1
            ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
            kernel = np.outer(gauss, gauss).astype(np.float32)
            return torch.from_numpy(kernel / np.sum(kernel)).unsqueeze(0).unsqueeze(1)

        self.gaussian_kernel = gaussian_kernel(2, 1.0).to(device)

        # Create crit
        self.crit = None
        if dist == 'l1':
            self.crit = torch.nn.L1Loss(reduction='none')
        elif dist == 'l2':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif dist == 'smoothl1':
            self.crit = torch.nn.SmoothL1Loss(reduction='none')
        elif dist == 'bce':
            self.crit = torch.nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown loss type: {dist}')

        self.reset()

    def cal_mask_from_filter(self, mpf_dots, dots_weight, rad=5):
        """
        :param mpf_dots: 正序  [T, 2, Kc]  # ..., [xc^t-2, yc], [xc^t-1, yc], [xc^t, yc]
        :param dots_weight: [1, 1, Kc]
        :return:
        """
        mask_set = []
        for frm_idx in range(0, self.T):
            mask = torch.zeros([self.H, self.W]).to(self.device)
            pf_dot = mpf_dots[frm_idx].squeeze().long()  # [2, Kc]
            x_coord, y_coord = pf_dot  # [Kc]
            value = dots_weight.view(-1)  # [Kc]
            mask = mask.index_put(indices=[y_coord, x_coord], values=value)
            mask = mask.view(1, 1, self.H, self.W)
            mask = torch.nn.functional.max_pool2d(input=mask, kernel_size=2 * rad + 1,
                                                  stride=1, padding=rad)
            mask[0, 0, :, :30] = 0.0
            mask_set.append(mask)
        return mask_set

    def reset(self):
        mu_start = torch.rand([self.num_pat]).to(self.device) * self.W
        sigma_start = self.args['max_sig2'] * torch.ones_like(mu_start)
        self.pat_distribution = torch.stack([mu_start, sigma_start], dim=1)  # [Kp, 2]

    def _warp_with_coord(self, pf_dot, src_mat, mode='bilinear'):
        """
        :param pf_dot: [C, K, 2]
        :param disp_mat: [1, 1, H, W]
        :return: disp [C, K]
        """
        if len(pf_dot.shape) == 2:
            pf_dot = pf_dot.unsqueeze(0)
        c = pf_dot.shape[0]
        pf_dot = pf_dot.clone().view(c, -1, 1, 2)  # [C, K, 1, 2]
        pf_dot = pf_dot.permute(2, 1, 0, 3)  # [1, K, C, 2]
        pf_dot[:, :, :, 0] = 2.0 * (pf_dot[:, :, :, 0] / (self.W - 1) - 0.5)
        pf_dot[:, :, :, 1] = 2.0 * (pf_dot[:, :, :, 1] / (self.H - 1) - 0.5)
        res_warp = F.grid_sample(src_mat.float(), pf_dot, mode=mode,  # [1, 1, K, C]
                                 padding_mode='border', align_corners=False)
        res_warp = res_warp.view(res_warp.shape[-2:]).permute(1, 0)  # [C, K]
        return res_warp

    @staticmethod
    def _average_with_mask(src_mat, mask_mat, dim, keepdim=True):
        valid_num = mask_mat.sum(dim=dim, keepdim=keepdim)
        avg = torch.sum(src_mat * mask_mat, dim=dim, keepdim=keepdim) / torch.clamp_min(valid_num, 1.0)
        return avg

    def _kalman_update(self, observation, mask_hit, distribution_dots):
        """
        Predict distribution:
            mu_prior = mu, sig2_prior = sig2 + sig2_prior_add
        :param distribution_dots: [Kc, 5]
        :return:
        """
        observation = observation.squeeze()  # [Kc, 2]
        mask_hit = mask_hit.squeeze()  # [Kc]

        # Prediction: mu_prior = mu, sig2_prior = sig2 + sig2_prior_add
        mu_prior = distribution_dots[:, 3]
        sig2_prior = torch.clamp_max(distribution_dots[:, 4] + self.args['sig2_prior_add'], self.args['max_sig2'])

        # Refine
        mu_check, sig2_check = observation[:, 0], observation[:, 1]
        kalman_gain = sig2_prior / (sig2_prior + sig2_check)  # [Kc]
        kalman_gain *= mask_hit
        mu_post = mu_prior + kalman_gain * (mu_check - mu_prior)
        sig2_post = sig2_prior - kalman_gain * sig2_prior

        return mu_post, sig2_post

    def forward(self, disp_list, mpf_dots, distribution_dots):
        """
            disp_list: [1, 1, H, W] * frm_num
            mpf_dots: [T, 2, Kc]  # ..., [xc^t-2, yc], [xc^t-1, yc], [xc^t, yc]
            distribution_dots: [Kc, 5], [id_cam, x^t-T, y^t-T, mu, sig2]  for each flow set
        """

        # 0. disp_list -> x coord in projector (xp)
        xp_set = []
        mask_set = []
        for frm_idx in range(0, self.T):
            disp_warp = self._warp_with_coord(mpf_dots[frm_idx].permute(1, 0), disp_list[frm_idx])  # [1, Kc]
            xp_dot = mpf_dots[frm_idx, 0, :] - disp_warp.view(-1)  # [Kc]
            xp_set.append(xp_dot)
            mask_valid = (mpf_dots[frm_idx, 0, :] >= 0).float()
            mask_set.append(mask_valid)
        xp_clip = torch.stack(xp_set, dim=0).view(self.T, 1, -1)  # [T, 1, Kc]
        mask_clip = torch.stack(mask_set, dim=0).view(self.T, 1, -1)  # [T, 1, Kc]
        xp_clip[:, 0, 0] = 0.0
        mask_clip[:, 0, 0] = 0.0  # [T, 1, Kc]

        with torch.no_grad():
            # 1. 计算对应的mu, sig；
            flow_len = mask_clip.sum(dim=0, keepdim=True)  # [1, 1, Kc]
            mask_clip[flow_len.repeat(self.T, 1, 1) <= 2.0] = 0.0
            mu_cam = self._average_with_mask(xp_clip, mask_clip, dim=0)  # [1, 1, Kc]
            sig2_cam = self._average_with_mask((xp_clip - mu_cam) ** 2, mask_clip, dim=0)
            # sig2_cam *= (2 - flow_len / self.T) * 2.0  # (2 - flow_len / self.T) * 0.5  # [0.5, 1.875/2 ~ 0.9]

            # 2. 根据mu进行相应的xp和id_p的对应
            xy_grid = torch.stack([mu_cam.view(-1), mpf_dots[0, 1, :]], dim=1).view(-1, 2)  # [Kc, 2]
            id_p = self._warp_with_coord(xy_grid, self.pat_info['pid'], mode='nearest').view(1, 1, -1)  # [1, 1, Kc]
            id_p[0, 0, 0] = 0

            # 3. 从id_p进行重合判断，并取最小sigma2以进行筛选。
            id_c = torch.arange(0, mu_cam.shape[2]).view(1, 1, -1).to(self.device)  # [1, 1, Kc]
            pt_cam = torch.stack([id_p, id_c, mu_cam, sig2_cam], dim=3)  # [1, 1, Kc, 4]
            _, sig2_idx = torch.sort(pt_cam[0, 0, :, 3], dim=0, descending=False)
            pt_cam = pt_cam[:, :, sig2_idx, :]
            pt_cam[0, 0, 0, :] = torch.Tensor([0.0, 0.0, 0.0, self.args['max_sig2']])

            # 4. 投影至projector空间；更新相应的mu与sigma（对有效部分
            pt_pro = torch.zeros([1, 1, self.num_pat, 4]).to(self.device)
            cam2pro_vec = pt_cam.squeeze()[:, 0].long()
            pt_pro[:, :, cam2pro_vec, :] = pt_cam  # [1, 1, Kp, 4]; id_p, id_c, mu, sigma
            pt_pro[0, 0, 0, :] = torch.Tensor([0.0, 0.0, 0.0, self.args['max_sig2']])

            # 5. 根据edge，判断其可靠性。（参考pointerlib
            # 5.1 edge_id_c
            id_c_src = pt_pro[0, 0, :, 1].long()  # [Kp]  id_c
            edge_id_p = self.pat_info['edge'].view(-1, 8)  # [Kp * 8]
            edge_id_c = torch.index_select(id_c_src, dim=0, index=edge_id_p.view(-1)).view(-1, 8)  # [Kp, 8]

            # 5.2 pt_pro_pos_c_diff_avg, [1, 2, Kp, 8]
            pt_pro_pos_c = torch.index_select(mpf_dots, dim=2, index=id_c_src).view(self.T, 2, -1, 1)  # [T, 2, Kp, 1]
            pt_pro_pos_cnbr = torch.index_select(mpf_dots, dim=2, index=edge_id_c.view(-1))
            pt_pro_pos_cnbr = pt_pro_pos_cnbr.view(self.T, 2, -1, 8)  # [T, 2, Kp, 8]
            pt_pro_mask_c = torch.index_select(mask_clip, dim=2, index=id_c_src).view(self.T, 1, -1, 1)  # [T, 1, Kp, 1]
            pt_pro_mask_cnbr = torch.index_select(mask_clip, dim=2, index=edge_id_c.view(-1))
            pt_pro_mask_cnbr = pt_pro_mask_cnbr.view(self.T, 1, -1, 8)
            pt_pro_mask_inter = pt_pro_mask_c * pt_pro_mask_cnbr  # [T, 1, Kp, 8]
            pt_pro_pos_c_diff = pt_pro_pos_c - pt_pro_pos_cnbr  # [T, 2, Kp, 8]
            pt_pro_pos_c_diff_avg = self._average_with_mask(pt_pro_pos_c_diff, pt_pro_mask_inter, dim=0)

            # 5.3 epe
            pt_pro_pos_diff = self.pat_info['diff'].view(1, 2, -1, 8)  # [1, 2, Kp, 8]
            pt_pro_edge_epe = torch.linalg.norm(pt_pro_pos_c_diff_avg - pt_pro_pos_diff,
                                                ord=1, dim=1, keepdim=True)  # [1, 1, Kp, 8]

            # 5.3 epe -> mask_edge, [1, 1, Kp, 8]
            mask_edge_c = (edge_id_c > 0).float()  # [Kp, 8]
            mask_edge_inter = (pt_pro_mask_inter.sum(dim=0, keepdim=True) > 0).float()  # [1, 1, Kp, 8]
            mask_edge = mask_edge_c * mask_edge_inter
            mask_edge[pt_pro_edge_epe > self.args['edge_thr']] = 0.0  # [1, 1, Kp, 8]
            mask_edge_all = (self.pat_info['edge'] > 0).float()  # [1, 1, Kp, 8]
            edge_weight = mask_edge_all.sum(dim=3) / torch.clamp_min(mask_edge.sum(dim=3), 1.0)  # [1, 1, Kp]

            # 5.4 adjust sig2
            pt_pro[0, 0, :, -1] *= self.args['alpha_edge'] * edge_weight.view(-1)

            # 6. 反投影回cam空间并更新。
            pt_cam_back = torch.zeros_like(pt_cam)  # [1, 1, Kc, 4], id_p, id_c, mu, sig2
            pro2cam_vec = pt_pro[0, 0, :, 1].long()  # [Kp]
            pt_cam_back[:, :, pro2cam_vec, :] = pt_pro
            pt_cam_back[0, 0, 0, :] = torch.Tensor([0.0, 0.0, 0.0, self.args['max_sig2']])
            mask_hit = (pt_cam_back[:, :, :, 0] > 0).float()  # [1, 1, Kc]
            mu_post, sig2_post = self._kalman_update(pt_cam_back[0, 0, :, 2:], mask_hit, distribution_dots)
            pt_cam_back[0, 0, :, 2] = mu_post
            pt_cam_back[0, 0, :, 3] = sig2_post
            new_distribution = distribution_dots.clone()
            new_distribution[:, 3] = mu_post
            new_distribution[:, 4] = sig2_post

            pt_cam_edge = torch.zeros(pt_cam.shape[2], 8).long().to(self.device)
            pt_cam_edge[pro2cam_vec, :] = (edge_id_c * mask_edge).long()

            # 7. 更新对应的weight
            mu_cam_back = pt_cam_back[0, 0, :, 2]
            xy_grid = torch.stack([mu_cam_back, mpf_dots[0, 1, :]], dim=1).view(-1, 2)  # [Kc, 2]
            xp_gt = self._warp_with_coord(xy_grid, self.pat_info['xp'], mode='nearest').view(1, 1, -1)  # [1, 1, Kc]
            xp_weight = torch.exp(-pt_cam_back[:, :, :, 3] / self.args['exp_sig2'])  # [1, 1, Kc]

        pass
        # 8. Cal err
        err_map = self.crit(xp_clip, xp_gt.detach().repeat(8, 1, 1)) * xp_weight  # [T, 1, Kc]
        mask_err = mask_clip
        val = (err_map * mask_err).sum() / (mask_err.sum() + 1e-8)

        return val, xp_weight, pt_cam_back, new_distribution, pt_cam_edge
