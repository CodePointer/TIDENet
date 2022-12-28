# -*- coding: utf-8 -*-

# - Package Imports - #
import numpy as np
import torch
import cv2
from pathlib import Path

import utils.pointerlib as plb
from utils.pattern_flow import PFlowEstimatorLK
from worker.worker import Worker
from tide.tide_net import TIDEFeature, TIDEHidden, TIDEUpdate, TIDEInit
from models.img_clip_dataset import ImgClipDataset
from models.supervise import SuperviseDistLoss
from models.layers import LCN, WarpLayer


# - Coding Part - #
class ExpTIDEWorker(Worker):
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
        self.pf_estimator = None
        self.warp_layer_dn8 = None

        self.last_frm = {}
        self.for_viz = {'frm_max': 128}

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """

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
        # self.networks['TIDE_Init'] = TIDEInit()
        # self.network_static_list.append('TIDE_Init')
        self.networks['TIDE_Ft'] = TIDEFeature()
        self.networks['TIDE_NtH'] = TIDEHidden()
        self.networks['TIDE_Up'] = TIDEUpdate(mask_flag=True, iter_times=1)
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
        self.warp_layer_dn8 = WarpLayer(self.imsize[0] // 8, self.imsize[1] // 8)
        self.pf_estimator = PFlowEstimatorLK(*self.imsize, win_rad=31)
        self.loss_funcs['dp-super'] = self.super_dist
        self.logging(f'Loss types: {self.loss_funcs.keys()}')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        # 1. Split
        key_list = list(data.keys())
        for key in key_list:
            if key in ['disp', 'mask']:
                data[key] = [x for x in torch.chunk(data[key], self.args.clip_len, dim=1)]
            elif key in ['img']:
                img_list = []
                std_list = []
                for img in torch.chunk(data[key], self.args.clip_len, dim=1):
                    img_lcn, img_std = self.lcn_layer(img)
                    img_list.append(torch.cat([img, img_lcn], dim=1))
                    std_list.append(img_std)
                data[key] = img_list
                data[f'{key}_std'] = std_list

        # PF
        data['pf'] = []
        for f in range(self.args.clip_len):
            if f == 0:
                if self.status in ['Train'] or data['frm_start'].item() == 0:
                    img_lst = data['img'][f]
                else:
                    img_lst = self.last_frm['img'].detach().cpu()
            else:
                img_lst = data['img'][f - 1]

            pf_mat, _ = self.pf_estimator.estimate_dn8(src_img=data['img'][f], dst_img=img_lst)
            pf_dp_mat = - pf_mat
            data['pf'].append(pf_dp_mat)

        # 2. Copy to device
        for f in range(self.args.clip_len):
            for key in ['img', 'pf', 'img_std', 'disp', 'mask']:
                if key in data and data[key][f] is not None:
                    data[key][f] = data[key][f].to(self.device)

        # pat
        batch = data['img'][0].shape[0]
        pat = self.pattern.repeat(batch, 1, 1, 1)
        pat_lcn, _ = self.lcn_layer(pat)
        data['pat'] = torch.cat([pat, pat_lcn], dim=1).to(self.device)

        return data

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        if self.status in ['Train']:  # Batch based.
            disp_outs = []

            with torch.no_grad():
                # disp = self.networks['TIDE_Init'](img=data['img'][0], pat=data['pat'])
                disp = torch.ones_like(data['img'][0][:, :1, :, :]) * 200.0
                disp_outs.append(disp)
            net_h = self.networks['TIDE_NtH'](img=data['img'][0])
            fmap_pat = self.networks['TIDE_Ft'](img=data['pat'])

            for frm_idx in range(0, self.args.clip_len):
                # Warp hidden & disp_lst from Pattern flow
                disp_lst = disp_outs[-1]
                pf_dp_mat = data['pf'][frm_idx]
                disp_pred = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=disp_lst) - pf_dp_mat
                net_h = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=net_h)

                # Estimate Disparity
                fmap_img = self.networks['TIDE_Ft'](data['img'][frm_idx])
                disps, net_h, _ = self.networks['TIDE_Up'](fmap_img, fmap_pat, data['img'][frm_idx],
                                                           flow_init=disp_pred / 8.0, h=net_h)
                disp_outs.append(disps[-1])

            disp_outs.pop(0)  # Ignore the very first frame

        else:  # Sequence based.
            disp_outs = []
            frm_start = data['frm_start'].item()
            self.for_viz['frm_start'] = int(frm_start)
            for frm_idx in range(self.args.clip_len):
                with torch.no_grad():
                    if frm_start == 0 and frm_idx == 0:  # Very first frame
                        disp_pred = torch.ones_like(data['img'][0][:, :1, :, :]) * 200.0
                        net_h = self.networks['TIDE_NtH'](img=data['img'][frm_idx])
                        self.last_frm['fmap_pat'] = self.networks['TIDE_Ft'](img=data['pat'])
                    else:
                        disp_lst = self.last_frm['disp'].detach()
                        pf_dp_mat = data['pf'][frm_idx]
                        disp_pred = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=disp_lst) - pf_dp_mat
                        net_h = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=self.last_frm['net_h']).detach()

                fmap_img = self.networks['TIDE_Ft'](img=data['img'][frm_idx])
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

        dp_super_loss = torch.zeros(1).to(self.device)
        for f in range(0, self.args.clip_len):
            disp_est = disps[f]
            mask = data['img_std'][f] if self.status == 'Train' else data['mask'][f]
            mask[data['disp'][f] == 0] = 0.0
            dp_super_loss += self.loss_record(
                'dp-super', pred=disp_est, target=data['disp'][f], mask=mask
            )
            total_loss += dp_super_loss

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
        data_idx = int(data['idx'].item())
        seq_folder, frm_start = dataset.samples[data_idx]
        seq_out_dir = out_dir / seq_folder.name
        for frm_idx in range(self.args.clip_len):
            disp_est = net_out[frm_idx][0]
            plb.imsave(seq_out_dir / 'disp' / f'disp_{frm_idx + frm_start}.png', disp_est,
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
        else:
            # Check frm_start & frm_end
            if kwargs['idx'] * self.args.clip_len <= self.for_viz['frm_max']:
                if kwargs['idx'] % 16 == 0:
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

        disp_outs = net_out

        # Visualize disparity & errors

        disp_gts = []
        for f in range(0, self.args.clip_len):
            disps = []
            disp_list = disp_outs[f]
            total_len = len(disp_list)
            disp_gt = data['disp'][f].detach().cpu()
            disp_gts.append(pvf.disp_visual(disp_gt, range_val=[0, max_disp]))

            disp_est = disp_list.detach().cpu()
            _, super_err = self.super_dist(pred=disp_est, target=disp_gt)
            mask = None if self.status == 'Train' else data['mask'][f].detach().cpu()
            disps.append(pvf.disp_visual(disp_est, range_val=[0, max_disp]))
            disps.append(pvf.err_visual(super_err, mask_mat=mask, max_val=max_dp_err, color_map=cv2.COLORMAP_HOT))

            disp_viz = pvf.img_concat(disps, total_len, 2)
            frm_idx = f if self.status == 'Train' else self.for_viz['frm_start'] + f
            self.loss_writer.add_image(f'{tag}/disp_est{frm_idx}', disp_viz, step)

        gt_viz = pvf.img_concat(disp_gts, self.args.clip_len, 1)
        self.loss_writer.add_image(f'{tag}/disp_rs', gt_viz, step)

        self.loss_writer.flush()
        pass
