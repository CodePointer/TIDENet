# -*- coding: utf-8 -*-

# - Package Imports - #
import numpy as np
import torch
import cv2

import utils.pointerlib as plb
from worker.worker import Worker
from tide.tide_net import RIDEInit
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
        self.imsize = None
        self.pattern = None
        self.super_dist = None
        self.lcn_layer = None

        self.for_viz = {'frm_max': 128}

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """
        train_folders = plb.subfolders(self.data_dir)
        test_folders = plb.subfolders(self.test_dir)
        train_paras = dict(clip_len=self.args.clip_len, frm_step=1, clip_jump=0, aug_flag=True)
        test_paras = dict(clip_len=self.args.clip_len, frm_step=1, clip_jump=0, blur=False)

        if self.args.exp_type in ['train']:
            self.save_flag = False
            assert train_folders is not None, '--train_dir is required for exp_type=train.'
        elif self.args.exp_type in ['eval']:
            train_folders = None
            self.save_flag = self.args.save_res
            assert test_folders is not None, '--test_dir is required for exp_type=eval.'
        else:
            raise NotImplementedError(f'Wrong exp_type: {self.args.loss_type}')

        self.train_dataset = None
        if train_folders is not None:
            self.train_dataset = ImgClipDataset(train_folders, pattern_path=self.data_dir / 'rect_pattern.png',
                                                **train_paras)
        self.test_dataset = []
        self.res_writers = []
        if test_folders is not None:
            self.test_dataset.append(ImgClipDataset(test_folders, pattern_path=self.test_dir / 'rect_pattern.png',
                                                    **test_paras))
            self.res_writers.append(self.create_res_writers(test_folders[0].parent))

        main_dataset = self.test_dataset[0] if self.args.exp_type == 'eval' else self.train_dataset
        self.imsize = main_dataset.get_size()
        self.pattern = main_dataset.get_pattern().unsqueeze(0)

        self.logging(f'--train_dir: {self.data_dir}')
        self.logging(f'--test_dir: {self.test_dir}')
        pass

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['InitNet'] = RIDEInit()
        if self.args.exp_type in ['eval']:
            self.network_static_list.append('InitNet')
        self.logging(f'--networks: {",".join(self.networks.keys())}')
        self.logging(f'--networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_dist = SuperviseDistLoss(dist='smoothl1')
        self.lcn_layer = LCN(radius=5, epsilon=1e-6)
        self.loss_funcs['dp-super'] = self.super_dist
        self.logging(f'--loss types: {self.loss_funcs.keys()}')

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
        frm_start = data['frm_start'].item()
        self.for_viz['frm_start'] = int(frm_start)
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

    def callback_after_train(self, epoch):
        """
            Set the save_flag for saving result.
        """
        if self.args.save_stone > 0 and (epoch + 1) % self.args.save_stone == 0:
            self.save_flag = True
        else:
            self.save_flag = False
        pass

    def callback_save_res(self, data, net_out, dataset, res_writer):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        out_dir, config = res_writer
        disp_outs = net_out
        data_idx = int(data['idx'].item())
        seq_folder, frm_start = dataset.samples[data_idx]

        # Save disp
        save_folder = out_dir / seq_folder.name / 'disp'
        save_folder.mkdir(parents=True, exist_ok=True)
        for frm_idx in range(self.args.clip_len):
            disp_est = disp_outs[frm_idx]
            disp_est = plb.t2a(disp_est)
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
        max_disp = 300.0

        disp_outs = net_out

        # 1. Visualize disparity & errors
        for f in range(0, self.args.clip_len):
            disp_gt = data['disp'][f].detach().cpu()
            disp_gt_viz = pvf.disp_visual(disp_gt, range_val=[0, max_disp])

            disp_est = disp_outs[f].detach().cpu()
            _, super_err = self.super_dist(pred=disp_est, target=disp_gt)

            mask = None if self.status == 'Train' else data['mask'][f].detach().cpu()
            disp_est_viz = pvf.disp_visual(disp_est, range_val=[0, max_disp])
            err_map_viz = pvf.err_visual(super_err, mask_mat=mask, max_val=max_dp_err, color_map=cv2.COLORMAP_HOT)

            disp_viz = pvf.img_concat([disp_gt_viz, disp_est_viz, err_map_viz], 1, 3)
            frm_idx = f if self.status == 'Train' else self.for_viz['frm_start'] + f
            self.loss_writer.add_image(f'{tag}/disp_est{frm_idx}', disp_viz, step)

        self.loss_writer.flush()
        pass
