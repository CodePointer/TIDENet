# -*- coding: utf-8 -*-

# - Package Imports - #
import numpy as np
import torch
import cv2
from pathlib import Path

import utils.pointerlib as plb
from worker.worker import Worker
from tide.tide_net import TIDEInit
from models.img_clip_dataset import ImgClipDataset
from models.supervise import SuperviseDistLoss
from models.layers import LCN


# - Coding Part - #
class ExpInitWorker(Worker):
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

        self.for_viz = {}

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
                clip_jump=0,
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

        if self.args.exp_type in ['train', 'online']:
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
        self.networks['InitNet'] = TIDEInit()
        if self.args.exp_type in ['eval']:
            self.network_static_list.append('InitNet')
        self.logging(f'Networks: {",".join(self.networks.keys())}')
        self.logging(f'Networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_dist = SuperviseDistLoss(dist='smoothl1')
        self.lcn_layer = LCN(radius=5, epsilon=1e-6)
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

        # 2. Copy to device
        for f in range(self.args.clip_len):
            data['img'][f] = data['img'][f].to(self.device)
            data['disp'][f] = data['disp'][f].to(self.device)
            data['mask'][f] = data['mask'][f].to(self.device)
            data['img_std'][f] = data['img_std'][f].to(self.device)

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
        disp_outs = []
        # frm_start = data['frm_start'].item()
        # self.for_viz['frm_start'] = int(frm_start)
        for frm_idx in range(0, self.args.clip_len):
            disp = self.networks['InitNet'](img=data['img'][frm_idx], pat=data['pat'])
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
        disp_outs = net_out
        data_idx = int(data['idx'].item())
        seq_folder, frm_start = dataset.samples[data_idx]
        dataset_tag = dataset.get_tag()
        out_dir = Path(self.args.out_dir) / self.args.res_dir_name / dataset_tag / f'epoch_{epoch:04}'

        # Save disp
        save_folder = out_dir / seq_folder.name / 'disp'
        save_folder.mkdir(parents=True, exist_ok=True)
        for frm_idx in range(self.args.clip_len):
            for n in range(self.args.batch_num):
                disp_est = disp_outs[frm_idx][n]
                frm_start = data[frm_idx][n].detach().cpu().item()
                plb.imsave(save_folder / f'disp_{frm_idx + frm_start}.png', disp_est,
                           scale=1e2, img_type=np.uint16)
        pass

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

        # 1. Visualize disparity & errors
        for f in range(0, self.args.clip_len):
            disp_gt = data['disp'][f][0].detach().cpu()
            disp_gt_viz = pvf.disp_visual(disp_gt, range_val=[0, max_disp])

            disp_est = disp_outs[f][0].detach().cpu()
            _, super_err = self.super_dist(pred=disp_est, target=disp_gt)

            mask = None if self.status == 'Train' else data['mask'][f][0].detach().cpu()
            disp_est_viz = pvf.disp_visual(disp_est, range_val=[0, max_disp])
            err_map_viz = pvf.err_visual(super_err, mask_mat=mask, max_val=max_dp_err, color_map=cv2.COLORMAP_HOT)

            disp_viz = pvf.img_concat([disp_gt_viz, disp_est_viz, err_map_viz], 1, 3)
            frm_idx = f if self.status == 'Train' else data['frm_start'][0].item() + f
            self.loss_writer.add_image(f'{tag}/disp_est{frm_idx}', disp_viz, step)

        self.loss_writer.flush()
        pass
