# -*- coding: utf-8 -*-

# - Package Imports - #
import numpy as np
import torch
import cv2

import utils.pointerlib as plb
from utils.pattern_flow import PFlowEstimatorLK, MPFlowEstimator
from worker.worker import Worker
from tide.tide_net import TIDEFeature, TIDEHidden, TIDEUpdate, TIDEInit
from models.img_clip_dataset import ImgClipDataset
from models.supervise import WarpedPhotoLoss, SuperviseDistLoss, PFDistLoss
from models.layers import LCN, WarpLayer


# - Coding Part - #
class ExpTIDEWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)
        self.imsize = None

        self.pattern = None
        self.pat_info = None
        self.mpf_distribution = None

        self.super_dist = None
        self.pf_dist = None
        self.warp_loss = None
        self.warp_layer = None
        self.warp_layer_dn8 = None
        self.lcn_layer = None
        self.pf_estimator = None
        self.mpf_estimator = None

        self.last_frm = {}
        self.for_viz = {'frm_max': 128}

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """

        # if self.args.exp_type == 'train':  # Pretrain
        #     self.save_flag = False
        #     assert train_folders is not None, '--train_dir is required for exp_type=train.'
        # elif self.args.exp_type == 'eval':  # Without BP: TIDE
        #     train_folders = None
        #     self.save_flag = self.args.save_res
        #     assert test_folders is not None, '--test_dir is required for exp_type=eval.'
        # elif self.args.exp_type == 'online':  # With BP for training: TIDE-Online
        #     train_paras['aug_flag'] = False
        #     self.train_shuffle = False
        #     test_folders = None
        #     self.save_flag = self.args.save_res
        #     assert train_folders is not None, '--train_dir is required for exp_type=online.'
        # else:
        #     raise NotImplementedError(f'Wrong exp_type: {self.args.exp_type}')

        train_folders = plb.subfolders(self.train_dir)
        self.train_dataset = None
        if train_folders is not None:
            self.train_dataset = ImgClipDataset(
                dataset_tag=self.train_dir.name,
                seq_folders=train_folders,
                pattern_path=self.train_dir / 'rect_pattern.png',
                clip_len=self.args.clip_len,
                frm_step=1,
                clip_jump=0,
                aug_flag=True
            )
        if self.args.exp_type == 'online':
            self.train_shuffle = False

        test_folders = plb.subfolders(self.test_dir)
        self.test_dataset = []
        if test_folders is not None:
            self.test_dataset.append(ImgClipDataset(
                dataset_tag=self.test_dir.name,
                seq_folders=test_folders,
                pattern_path=self.test_dir / 'rect_pattern.png',
                clip_len=self.args.clip_len,
                frm_step=1,
                clip_jump=0,
                blur=False
            ))

        main_dataset = self.test_dataset[0] if self.args.exp_type == 'eval' else self.train_dataset
        self.imsize = main_dataset.get_size()
        self.pattern = main_dataset.get_pattern().unsqueeze(0)
        self.pat_info = main_dataset.get_pat_info()
        self.pat_info = {x: self.pat_info[x].to(self.device) for x in self.pat_info}

        self.logging(f'--train_dir: {self.train_dir}')
        self.logging(f'--test_dir: {self.test_dir}')
        pass

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['TIDE_Init'] = TIDEInit()
        self.network_static_list.append('TIDE_Init')
        self.networks['TIDE_Ft'] = TIDEFeature()
        self.networks['TIDE_NtH'] = TIDEHidden()
        self.networks['TIDE_Up'] = TIDEUpdate(mask_flag=True, iter_times=1)
        self.logging(f'--networks: {",".join(self.networks.keys())}')
        self.logging(f'--networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_dist = SuperviseDistLoss(dist='smoothl1')
        self.warp_layer = WarpLayer(*self.imsize)
        self.warp_layer_dn8 = WarpLayer(self.imsize[0] // 8, self.imsize[1] // 8)
        self.warp_loss = WarpedPhotoLoss(*self.imsize, dist='smoothl1')
        self.pf_dist = PFDistLoss(self.pat_info, self.args.clip_len, *self.imsize, device=self.device)
        self.lcn_layer = LCN(radius=9, epsilon=1e-6)

        if self.args.loss_type == 'su':
            self.loss_funcs['dp-super'] = self.super_dist
        elif self.args.loss_type == 'ph':
            self.loss_funcs['dp-ph'] = self.warp_loss
        elif self.args.loss_type == 'pf':
            self.loss_funcs['dp-ph'] = self.warp_loss
            self.loss_funcs['dp-pf'] = self.pf_dist
        else:
            raise ValueError(f'Invalid exp_tag: {self.args.exp_tag}')

        self.pf_estimator = PFlowEstimatorLK(*self.imsize)
        self.mpf_estimator = MPFlowEstimator(self.args.clip_len, device=self.device)

        self.logging(f'--loss types: {self.loss_funcs.keys()}')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        # 1. Split
        key_list = list(data.keys())
        for key in key_list:
            if key in ['disp', 'mask', 'center', 'pf_dot']:
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
            data['img'][f] = data['img'][f].to(self.device)
            data['pf'][f] = data['pf'][f].to(self.device) if data['pf'][f] is not None else None
            data['center'][f] = data['center'][f].to(self.device)
            data['img_std'][f] = data['img_std'][f].to(self.device)
            if self.args.loss_type in ['su']:
                data['disp'][f] = data['disp'][f].to(self.device)
                data['mask'][f] = data['mask'][f].to(self.device)

        # 3. Multiple pf
        if data['frm_start'].item() == 0:
            self.mpf_distribution = None

        for f in range(self.args.clip_len):
            data['center'][f][data['disp'][f] == 0] = 0
        data['mpf'], self.mpf_distribution = self.mpf_estimator.run(data['center'], self.mpf_distribution)

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

            if self.args.exp_type == 'online' and data['frm_start'].item() > 0:
                disp_outs.append(self.last_frm['disp'])
                net_h = self.last_frm['net_h']
            else:
                with torch.no_grad():
                    disp = self.networks['TIDE_Init'](img=data['img'][0], pat=data['pat'])
                    disp_outs.append(disp)
                net_h = self.networks['TIDE_NtH'](img=data['img'][0])

            fmap_pat = self.networks['TIDE_Ft'](img=data['pat'])

            for frm_idx in range(0, self.args.clip_len):
                # Warp hidden & disp_lst from Pattern flow
                disp_lst = disp_outs[-1]
                pf_dp_mat = data['pf'][frm_idx]
                disp_pred = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=disp_lst) - pf_dp_mat
                net_h = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=net_h)

                # Estimate raft
                fmap_img = self.networks['TIDE_Ft'](data['img'][frm_idx])
                disps, net_h, _ = self.networks['TIDE_Up'](fmap_img, fmap_pat, data['img'][frm_idx],
                                                           flow_init=disp_pred / 8.0, h=net_h)
                disp_outs.append(disps[-1])

            if self.args.exp_type == 'online':
                self.last_frm['disp'] = disp_outs[-1]
                self.last_frm['img'] = data['img'][-1]
                self.last_frm['net_h'] = net_h.detach()

            disp_outs.pop(0)

        else:  # Sequence based.
            disp_outs = []
            frm_start = data['frm_start'].item()
            self.for_viz['frm_start'] = int(frm_start)
            for frm_idx in range(self.args.clip_len):
                if frm_start == 0 and frm_idx == 0:  # Very first frame
                    with torch.no_grad():
                        disp = self.networks['TIDE_Init'](img=data['img'][frm_idx], pat=data['pat'])
                        self.last_frm['net_h'] = self.networks['TIDE_NtH'](img=data['img'][frm_idx])
                        self.last_frm['fmap_pat'] = self.networks['TIDE_Ft'](img=data['pat'])
                else:
                    with torch.no_grad():
                        fmap_img = self.networks['TIDE_Ft'](img=data['img'][frm_idx])
                        disp_lst = self.last_frm['disp'].detach()
                        pf_dp_mat = data['pf'][frm_idx]
                        disp_pred = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=disp_lst) - pf_dp_mat
                        net_h = self.warp_layer_dn8(disp_mat=pf_dp_mat / 8.0, src_mat=self.last_frm['net_h']).detach()
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
                mask = data['img_std'][f] if self.status == 'Train' else data['mask'][f]
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
                mask = data['img_std'][f] if self.status == 'Train' else data['mask'][f]
                mask[data['disp'][f] == 0] = 0.0
                dp_ph_loss += self.loss_record(
                    'dp-ph', img_dst=data['img'][f][:, :1], img_src=data['pat'][:, :1],
                    disp_mat=disp, mask=mask, std=data['img_std'][f]
                )
            total_loss += dp_ph_loss

        elif self.args.loss_type == 'pf':
            dp_pf_loss, xp_weight, pt_cam_back, new_distribution, pt_cam_edge = self.loss_record(
                'dp-pf', disp_list=disps, mpf_dots=data['mpf'],
                distribution_dots=self.mpf_distribution, return_val_only=False
            )
            self.for_viz['pt_cam_back'] = pt_cam_back
            self.for_viz['xp_weight'] = xp_weight
            self.for_viz['pt_cam_edge'] = pt_cam_edge
            self.mpf_distribution = new_distribution

            if not torch.isnan(dp_pf_loss):
                total_loss += dp_pf_loss

            alpha_ph = 0.1
            dp_ph_loss = torch.zeros(1).to(self.device)
            mask_set = self.loss_funcs['dp-pf'].cal_mask_from_filter(data['mpf'], xp_weight, rad=3)
            self.for_viz['mask_set'] = mask_set
            for f in range(0, self.args.clip_len):
                disp = disps[f]
                mask = mask_set[f]  # * data['img_std'][f]
                loss_val, img_wrp, img_err, _, _ = self.loss_record(
                    'dp-ph', img_dst=data['img'][f][:, 1:], img_src=data['pat'][:, 1:],
                    disp_mat=disp, mask=mask, std=data['img_std'][f], return_val_only=False
                )
                dp_ph_loss += loss_val
            if not torch.isnan(dp_ph_loss):
                total_loss += alpha_ph * dp_ph_loss

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        pass

    def check_save_res(self, epoch):
        save_stone_flag = super().check_save_res(epoch)

        if self.args.exp_type in ['train', 'eval'] and self.status == 'Eval':
            save_status_flag = True
        elif self.args.exp_type in ['online'] and self.status == 'Train':
            save_status_flag = True
        else:
            save_status_flag = False

        return save_stone_flag and save_status_flag

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
        max_disp = 300.0

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
