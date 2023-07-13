# -*- coding: utf-8 -*-

# - Package Imports - #
import numpy as np
import torch
import cv2
from pathlib import Path

import utils.pointerlib as plb
from utils.pattern_flow import PFlowEstimatorLK, MPFlowEstimator
from worker.worker import Worker
from tide.tide_net import TIDEFeature, TIDEHidden, TIDEUpdate, TIDEInit
from tide.exp_tide import ExpTIDEWorker
from models.img_clip_dataset import ImgClipDataset
from models.supervise import WarpedPhotoLoss, SuperviseDistLoss, PFDistLoss
from models.layers import LCN, WarpLayer


# - Coding Part - #
class ExpOADEWorker(ExpTIDEWorker):
    """Online Adaptive Disparity Estimation (OADE)"""
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # # Exists parameters:
        # self.imsize = None
        # self.pattern = None
        # self.super_dist = None
        # self.lcn_layer = None
        # self.pf_estimator = None
        # self.warp_layer_dn8 = None
        # self.last_frm = {}
        # self.for_viz = {'frm_max': 128}

        self.flush_model = False
        self.pat_info = None
        self.mpf_distribution = None

        self.pf_dist = None
        self.warp_loss = None
        self.warp_layer = None
        self.mpf_estimator = None

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """
        assert self.args.exp_type == 'online', f'exp_type=online is needed for oade.'
        assert self.args.train_dir != '', f'train_dir is required for exp_type=online.'

        if self.args.frm_first >= 0:
            self.flush_model = True
        else:
            self.flush_model = False
            self.args.frm_first = 0

        train_folder = Path(self.args.train_dir)
        train_paras = dict(
            dataset_tag=train_folder.name,
            data_folder=train_folder,
            clip_len=self.args.clip_len,
            frm_first=self.args.frm_first,
            frm_step=1,
            clip_jump=0,
            blur=False,
            aug_flag=False,
        )
        self.train_dataset = ImgClipDataset(**train_paras)

        self.train_shuffle = False  # For online
        self.imsize = self.train_dataset.get_size()
        self.pattern = self.train_dataset.get_pattern().unsqueeze(0)
        self.pat_info = self.train_dataset.get_pat_info()
        self.pat_info = {x: self.pat_info[x].to(self.device) for x in self.pat_info}
        pass

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        super().init_networks()

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_dist = SuperviseDistLoss(dist='smoothl1')
        self.lcn_layer = LCN(radius=11, epsilon=1e-6)
        self.warp_layer_dn8 = WarpLayer(self.imsize[0] // 8, self.imsize[1] // 8)
        self.pf_estimator = PFlowEstimatorLK(*self.imsize, win_rad=31)

        self.warp_loss = WarpedPhotoLoss(*self.imsize, dist='smoothl1')
        self.pf_dist = PFDistLoss(self.pat_info, self.args.clip_len, *self.imsize, device=self.device)
        self.warp_layer = WarpLayer(*self.imsize)
        self.mpf_estimator = MPFlowEstimator(self.args.clip_len, device=self.device)

        if self.args.loss_type == 'su':
            self.loss_funcs['dp-super'] = self.super_dist
        elif self.args.loss_type == 'ph':
            self.loss_funcs['dp-ph'] = self.warp_loss
        elif self.args.loss_type in ['pf', 'pfwom']:
            self.loss_funcs['dp-pf'] = self.pf_dist
        elif self.args.loss_type in ['phpfwm', 'phpfwom']:
            self.loss_funcs['dp-ph'] = self.warp_loss
            self.loss_funcs['dp-pf'] = self.pf_dist
        else:
            raise ValueError(f'Invalid exp_tag: {self.args.exp_tag}')

        self.logging(f'Loss types: {self.loss_funcs.keys()}')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        data = super().data_process(idx, data)

        # Blob center
        assert 'center' in list(data.keys()), f'center data is required for exp_type=online.'

        # 1. Split rest
        data['center'] = [x for x in torch.chunk(data['center'], self.args.clip_len, dim=1)]

        # 2. Copy to device
        for f in range(self.args.clip_len):
            data['center'][f] = data['center'][f].to(self.device)

        # 3. mpf part
        if data['frm_start'].item() == self.args.frm_first:
            self.mpf_distribution = None
        data['mpf'], self.mpf_distribution = self.mpf_estimator.run(data['center'], self.mpf_distribution)

        return data

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        disp_outs = []
        frm_start = data['frm_start'].item()
        self.for_viz['frm_start'] = int(frm_start)

        # Flush model or not
        if self.flush_model and frm_start == self.args.frm_first:
            self._net_load(self.args.epoch_start - 1)

        # First frame
        for frm_idx in range(self.args.clip_len):
            with torch.no_grad():
                if frm_start == self.args.frm_first and frm_idx == 0:
                    disp_lst = torch.ones_like(data['img'][0][:, :1, :, :]) * 200.0
                    self.last_frm['net_h'] = self.networks['TIDE_NtH'](img=data['img'][frm_idx])
                    self.last_frm['fmap_pat'] = self.networks['TIDE_Ft'](img=data['pat'])
                else:
                    try:
                        disp_lst = self.last_frm['disp'].detach()
                    except KeyError as e:
                        print(frm_start, frm_idx, list(self.last_frm.keys()))
                        raise e
                pf_dp_mat = data['pf'][frm_idx]
                disp_pred = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=disp_lst) - pf_dp_mat
                net_h = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=self.last_frm['net_h']).detach()

            # Iteration
            fmap_img = self.networks['TIDE_Ft'](data['img'][frm_idx])
            disps, net_h, _ = self.networks['TIDE_Up'](fmap_img, self.last_frm['fmap_pat'], 
                                                       data['img'][frm_idx], flow_init=disp_pred / 8.0,
                                                       h=net_h)
            self.last_frm['net_h'] = net_h
            disp = disps[0]
            self.last_frm['disp'] = disp
            self.last_frm['img'] = data['img'][frm_idx]
            disp_outs.append(disp)

        return disp_outs

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        disps = net_out
        total_loss = torch.zeros(1).to(self.device)

        if self.args.loss_type == 'su':
            dp_super_loss = torch.zeros(1).to(self.device)
            for f in range(0, self.args.clip_len):
                disp_est = disps[f]
                mask = data['img_std'][f]
                # mask = data['center'][f]
                mask[data['disp'][f] == 0] = 0.0
                dp_super_loss += self.loss_record(
                    'dp-super', pred=disp_est, target=data['disp'][f], mask=mask
                )
            total_loss += dp_super_loss

        elif self.args.loss_type == 'ph':
            dp_ph_loss = torch.zeros(1).to(self.device)
            for f in range(0, self.args.clip_len):
                disp = disps[f]
                mask = data['img_std'][f]
                dp_ph_loss += self.loss_record(
                    'dp-ph', img_dst=data['img'][f][:, :1], img_src=data['pat'][:, :1],
                    disp_mat=disp, mask=mask
                )
            total_loss += dp_ph_loss

        elif self.args.loss_type in ['pf', 'pfwom', 'phpfwm', 'phpfwom']:
            weight_flag = False if self.args.loss_type == 'pfwom' else True
            dp_pf_loss, xp_weight, pt_cam_back, new_distribution, pt_cam_edge = self.loss_record(
                'dp-pf', disp_list=disps, mpf_dots=data['mpf'],
                distribution_dots=self.mpf_distribution, weight_flag=weight_flag, return_val_only=False
            )
            self.for_viz['pt_cam_back'] = pt_cam_back
            self.for_viz['xp_weight'] = xp_weight
            self.for_viz['pt_cam_edge'] = pt_cam_edge
            self.mpf_distribution = new_distribution

            if not torch.isnan(dp_pf_loss):
                total_loss += dp_pf_loss

            if self.args.loss_type in ['phpfwm', 'phpfwom']:
                alpha_ph = 0.1
                dp_ph_loss = torch.zeros(1).to(self.device)
                if self.args.loss_type == 'phpfwm':
                    mask_set = self.loss_funcs['dp-pf'].cal_mask_from_filter(data['mpf'], xp_weight, rad=3)
                else:  # phpfwom
                    mask_set = data['img_std']
                self.for_viz['mask_set'] = mask_set
                for f in range(0, self.args.clip_len):
                    disp = disps[f]
                    mask = mask_set[f]  # * data['img_std'][f]
                    loss_val, img_wrp, img_err, _, _ = self.loss_record(
                        'dp-ph', img_dst=data['img'][f][:, 1:], img_src=data['pat'][:, 1:],
                        disp_mat=disp, mask=mask, return_val_only=False
                    )
                    dp_ph_loss += loss_val
                if not torch.isnan(dp_ph_loss):
                    total_loss += alpha_ph * dp_ph_loss

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_save_res(self, epoch, data, net_out, dataset):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        super().callback_save_res(epoch, data, net_out, dataset)  # Save Disparity

        # dataset_tag = dataset.get_tag()
        # out_dir = self.res_dir / 'output' / dataset_tag / f'epoch_{epoch:05}'
        # out_dir.mkdir(parents=True, exist_ok=True)

        # Mask
        # for frm_idx in range(self.args.clip_len):
        #     mask_mpf = self.for_viz['mask_set'][frm_idx][0]
        #     plb.imsave(seq_out_dir / 'mask_mpf' / f'mask_{frm_idx + frm_start}.png', mask_mpf, mkdir=True)

        pass

    def check_img_visual(self, **kwargs):
        """
            The img_visual callback entries.
            Modified this function can control the report strategy during training and testing.
            (Additional)
        """
        super().check_img_visual(**kwargs)

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        super().callback_img_visual(data, net_out, tag, step)
        pass
