# -*- coding: utf-8 -*-
# @Description:
#   Worker for training connecting-the-dots.
#   From: https://github.com/autonomousvision/connecting_the_dots.
#   Only supervised training version is applied.

# - Package Imports - #
import torch
import numpy as np
from pathlib import Path

import utils.pointerlib as plb
from worker.worker import Worker
from models.supervise import SuperviseDistLoss
from models.img_clip_dataset import ImgClipDataset

from cmp.ctd.networks import DispEdgeDecoders, LCN


# - Coding Part - #
class ExpCtdWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # init_dataset():
        self.pattern = None
        self.imsizes = None

        # init_networks():
        self.channels_in = 2
        self.max_disp = 350
        self.output_ms = True

        # init_losses():
        self.lcn_in = None

        self.for_viz = {
            'frm_viz': [0, 255]
        }

    def init_dataset(self, **kwargs):
        """
            Requires:
                self.train_dataset
                self.test_dataset
                self.res_writers
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
        hei, wid = main_dataset.get_size()
        self.imsizes = [(hei, wid)]
        for iter in range(3):
            self.imsizes.append((int(self.imsizes[-1][0] / 2), int(self.imsizes[-1][1] / 2)))
        self.pattern = main_dataset.get_pattern().unsqueeze(0)
        pass

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['CTD'] = DispEdgeDecoders(channels_in=self.channels_in,
                                                max_disp=self.max_disp,
                                                imsizes=self.imsizes,
                                                output_ms=self.output_ms)

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        # Create LCN layer
        self.lcn_in = LCN(radius=5, epsilon=0.05)
        # self.lcn_in = self.lcn_in.to(self.device)

        # Create losses
        if self.args.exp_type == 'train':
            loss_func = SuperviseDistLoss(dist='smoothl1')
            self.loss_funcs = {f'Supervise{1 / 2 ** i}': loss_func for i in range(len(self.imsizes))}
            # self.loss_funcs['EdgeLoss'] = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1]).to(self.device))

        elif self.args.exp_type == 'eval':
            self.loss_funcs = dict()

        else:
            raise NotImplementedError(f'Invalid exp_type: {self.args.exp_type}')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        assert self.args.clip_len == 1

        # 1. img -> img_lcn
        img_lcn, img_std = self.lcn_in(data['img'])
        data['img'] = torch.cat([img_lcn, data['img']], dim=1)      # [N, 2, H, W]
        data['img_std'] = img_std

        # 2. Copy to device
        data['img'] = data['img'].to(self.device)
        data['disp'] = data['disp'].to(self.device)
        data['img_std'] = data['img_std'].to(self.device)

        return data

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """

        disp, edge = self.networks['CTD'](data['img'])
        return disp, edge

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        out, edge = net_out
        total_loss = torch.zeros(1).to(self.device)

        dp_super_loss = torch.zeros(1).to(self.device)
        for i, es in enumerate(out):
            loss_name = f'Supervise{1 / 2 ** i}'
            gt_dn = torch.nn.functional.interpolate(data['disp'], es.shape[-2:])
            dp_super_loss += self.loss_record(
                loss_name, pred=es, target=gt_dn
            )
        total_loss += dp_super_loss

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

        # if self.args.exp_type == 'train':
        #     out, edge = net_out
        #
        #     # edge loss
        #     for i, e in enumerate(edge):
        #         # inversed ground truth edge where 0 means edge
        #         grad = data[f'grad{i}'] < 0.2
        #         grad = grad.to(torch.float32)
        #         ids = data['id']
        #         mask = ids > self.train_edge
        #         if mask.sum() > 0:
        #             val = self.loss_funcs['EdgeLoss'](e[mask], grad[mask])
        #             self.avg_meters['EdgeLoss'].update(val, self.N)
        #         else:
        #             val = torch.zeros_like(total_loss[0])
        #         total_loss.append(val)
        #
        #     loss = sum(total_loss)
        #     self.avg_meters['Total'].update(loss, self.N)
        #
        #     return loss
        #
        # else:
        #     raise NotImplementedError(f'Invalid exp_type: {self.args.exp_type}')

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
        for n in range(self.args.batch_num):
            data_idx = int(data['idx'][n].item())
            seq_folder, frm_start = dataset.samples[data_idx]
            seq_out_dir = out_dir / seq_folder.name
            disp_ref = net_out[0][0]
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
        max_disp = self.max_disp

        disp_outs = net_out[0]
        batch_num = disp_outs[0].shape[0]
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
            disp_gt = data['disp'][n].detach().cpu()
            disp_viz_set = [
                pvf.disp_visual(disp_gt, range_val=[0, max_disp])
            ]
            for i, es in enumerate(disp_outs):
                disp_est = es[n].detach().cpu()
                es_up = torch.nn.functional.interpolate(disp_est[None], disp_gt.shape[-2:],
                                                        mode='nearest')[0]
                disp_viz_set.append(pvf.disp_visual(es_up, range_val=[0, max_disp]))

            viz_map = pvf.img_concat(disp_viz_set, len(disp_viz_set), 1)
            self.loss_writer.add_image(f'{tag}/{seq_tag}disp_est{frm_start}', viz_map, step)

        self.loss_writer.flush()
