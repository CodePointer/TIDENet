# -*- coding: utf-8 -*-

# @Time:      2021/03/04 16:53
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      exp_asn.py
# @Software:  VSCode
# @Description:
#   Empty worker for new workers writing.

# - Package Imports - #
import numpy as np
import torch
import cv2
from pathlib import Path

import utils.pointerlib as plb
from worker.worker import Worker
from models.img_clip_dataset import ImgClipDataset
from models.supervise import SuperviseDistLoss, WarpedPhotoLoss
from models.layers import LCN
from cmp.asn.ActiveStereoNet import ActiveStereoNet


# - Coding Part - #
class ExpASNWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)
        
        # init_dataset
        self.imsize = None
        self.pattern = None
        
        # init_losses
        self.super_dist = None
        self.lcn_layer = None
        self.warp_loss = None
        
        self.for_viz = {
            'frm_viz': [0, 255],  # Frm number for visualization
        }
        

    def init_dataset(self, **kwargs):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """
        assert self.args.clip_len == 1

        self.train_dataset = None
        if self.args.train_dir != '':
            train_folder = Path(self.args.train_dir)
            train_paras = dict(
                dataset_tag=train_folder.name,
                data_folder=train_folder,
                clip_len=self.args.clip_len,
                frm_step=1,
                clip_jump=32,
                blur=False,
                aug_flag=True,
            )
            self.train_dataset = ImgClipDataset(**train_paras)

        self.test_dataset = None
        if self.args.test_dir != '':
            test_folder = Path(self.args.test_dir)
            test_paras = dict(
                dataset_tag=test_folder.name,
                data_folder=test_folder,
                clip_len=self.args.clip_len,
                frm_step=1,
                clip_jump=0,
                blur=False,
                aug_flag=False,
            )
            self.test_dataset = ImgClipDataset(**test_paras)

        if self.args.exp_type in ['train']:
            assert self.train_dataset is not None, f'train_dir is required for exp_type={self.args.exp_type}.'
        elif self.args.exp_type in ['eval']:
            assert self.test_dataset is not None, f'test_dir is required for exp_type={self.args.exp_type}.'
        else:
            raise NotImplementedError(f'Wrong exp_type: {self.args.exp_type}')

        main_dataset = self.test_dataset if self.args.exp_type == 'eval' else self.train_dataset
        self.imsize = main_dataset.get_size()
        self.pattern = main_dataset.get_pattern().unsqueeze(0)
        pass

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['ASN'] = ActiveStereoNet(
            max_disp=400,
            bias_disp=100,
        )
        self.logging(f'Networks: {",".join(self.networks.keys())}')
        self.logging(f'Networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_dist = SuperviseDistLoss(dist='smoothl1')
        self.lcn_layer = LCN(radius=11, epsilon=1e-6)
        self.warp_loss = WarpedPhotoLoss(*self.imsize, dist='smoothl1')
        
        if self.args.loss_type == 'su':
            self.loss_funcs['dp-super-s'] = self.super_dist
            self.loss_funcs['dp-super-r'] = self.super_dist
        elif self.args.loss_type == 'ph':
            self.loss_funcs['dp-ph-s'] = self.warp_loss
            self.loss_funcs['dp-ph-r'] = self.warp_loss
        else:
            raise ValueError(f'Invalid exp_tag: {self.args.exp_tag}')

        self.logging(f'Loss types: {self.loss_funcs.keys()}')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        # 1. Split
        key_list = list(data.keys())
        for key in key_list:
            if key in ['img']:
                _, img_std = self.lcn_layer(data['img'])
                data[f'{key}_std'] = img_std
        
        # 2. Copy to device
        for key in ['img', 'img_std', 'disp', 'mask']:
            if key in data and data[key] is not None:
                data[key] = data[key].to(self.device)

        # pat
        batch = data['img'].shape[0]
        data['pat'] = self.pattern.repeat(batch, 1, 1, 1).to(self.device)
        
        return data

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        disp_sps, disp_ref = self.networks['ASN'](data['img'], data['pat'])
        return disp_sps, disp_ref

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        disp_sps, disp_ref = net_out
        total_loss = torch.zeros(1).to(self.device)

        if self.args.loss_type == 'su':
            dp_super_loss = torch.zeros(1).to(self.device)
            mask = data['img_std'] if self.status == 'Train' else data['mask']
            mask[data['disp'] == 0] = 0.0
            dp_super_loss += self.loss_record(
                'dp-super-s', pred=disp_sps, target=data['disp'], mask=mask
            )
            dp_super_loss += self.loss_record(
                'dp-super-r', pred=disp_ref, target=data['disp'], mask=mask
            )
            total_loss += dp_super_loss
            
        elif self.args.loss_type == 'ph':
            dp_ph_loss = torch.zeros(1).to(self.device)
            mask = data['img_std']
            dp_ph_loss += self.loss_record(
                'dp-ph-s', img_dst=data['img'], img_src=data['pat'],
                disp_mat=disp_sps, mask=mask
            )
            dp_ph_loss += self.loss_record(
                'dp-ph-r', img_dst=data['img'], img_src=data['pat'],
                disp_mat=disp_ref, mask=mask
            )
            total_loss += dp_ph_loss
        else:
            raise ValueError(f'Invalid exp_tag: {self.args.exp_tag}')

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_save_res(self, epoch, data, net_out, dataset):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        dataset_tag = dataset.get_tag()
        out_dir = self.res_dir / 'output' / dataset_tag / f'epoch_{epoch:05}'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Disparity
        batch_num = data['idx'].shape[0]
        for n in range(batch_num):
            data_idx = int(data['idx'][n].item())
            seq_folder, frm_start = dataset.samples[data_idx]
            seq_out_dir = out_dir / seq_folder.name
            disp_ref = net_out[0][n]
            plb.imsave(seq_out_dir / 'disp' / f'disp_{frm_start}.png', disp_ref,
                       scale=1e2, img_type=np.uint16, mkdir=True)
        pass

    def check_img_visual(self, **kwargs):
        """
            The img_visual callback entries.
            Modified this function can control the report strategy during training and testing.
            (Additional)
        """
        if self.status == 'Train':
            return super().check_img_visual(**kwargs)
        else:  # self.status == 'Eval'
            if self.loss_writer is None:
                return False
            if self.args.debug_mode:
                return True

            # Check frm_start & frm_end
            batch_num = kwargs['data']['frm_start'].shape[0]
            for n in range(batch_num):
                frm_start = kwargs['data']['frm_start'][n].detach().cpu().item()
                if frm_start in self.for_viz['frm_viz']:
                    return True
            return False

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        
        pvf = plb.VisualFactory
        max_dp_err = 10.0
        max_disp = 400.0
        
        disp_sps, disp_ref = net_out

        batch_num = disp_sps.shape[0]
        for n in range(batch_num):
            seq_tag = ''
            frm_start = data['frm_start'][n].detach().cpu().item()
            if self.status == 'Train':
                if n > 0:
                    continue
            else:  # self.status == 'Eval'
                if frm_start not in self.for_viz['frm_viz']:
                    continue
                data_idx = int(data['idx'][n].item())
                seq_folder, _ = self.test_dataset.samples[data_idx]
                seq_tag = f'{seq_folder.name}-'
                
            # Visualize disparity & errors
            disp_sps = net_out[0][n].detach().cpu()
            disp_ref = net_out[1][n].detach().cpu()
            disp_gt = data['disp'][n].detach().cpu()
            _, sps_err = self.super_dist(pred=disp_sps[None, :], target=disp_gt[None, :])
            _, ref_err = self.super_dist(pred=disp_ref[None, :], target=disp_gt[None, :])
            mask = None if self.status == 'Train' else data['mask'][n].detach().cpu()
            
            viz_map = pvf.img_concat(
                [pvf.disp_visual(disp_sps, range_val=[0, max_disp]),
                 pvf.err_visual(sps_err, mask, max_dp_err, color_map=cv2.COLORMAP_HOT),
                 pvf.disp_visual(disp_gt, range_val=[0, max_disp]),
                 pvf.disp_visual(disp_ref, range_val=[0, max_disp]),
                 pvf.err_visual(ref_err, mask, max_dp_err, color_map=cv2.COLORMAP_HOT),
                 pvf.disp_visual(disp_gt, range_val=[0, max_disp])],
                3, 2
            )
            
            self.loss_writer.add_image(f'{tag}/{seq_tag}disp_est{frm_start}', viz_map, step)

        self.loss_writer.flush()
        pass
