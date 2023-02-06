# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import cv2
import numpy as np
from configparser import ConfigParser
import utils.pointerlib as plb
from models.base_dataset import BaseDataset


# - Coding Part - #
def augment_images(imgs, rng, max_blur=2.5, max_noise=10.0, max_sp_noise=0.001):
    min_balance, max_balance = rng.uniform(0, 0.1), rng.uniform(0.8, 1.0)
    for i in range(len(imgs)):
        # get min/max values of image
        min_val = np.min(imgs[i])
        max_val = np.max(imgs[i])

        # init augmented image
        img_aug = imgs[i]

        # gaussian smoothing
        if rng.uniform(0, 1) < 0.5:
            img_aug = cv2.GaussianBlur(img_aug, (5, 5), rng.uniform(0.5, max_blur))

        # per-pixel gaussian noise
        img_aug = img_aug + rng.randn(*img_aug.shape) * rng.uniform(0.0, max_noise) / 255.0

        # salt-and-pepper noise
        if rng.uniform(0, 1) < 0.5:
            ratio = rng.uniform(0.0, max_sp_noise)
            img_shape = img_aug.shape
            img_aug = img_aug.flatten()
            coord = rng.choice(np.size(img_aug), int(np.size(img_aug) * ratio))
            img_aug[coord] = max_val
            coord = rng.choice(np.size(img_aug), int(np.size(img_aug) * ratio))
            img_aug[coord] = min_val
            img_aug = np.reshape(img_aug, img_shape)

        # clip intensities back to [0,1]
        img_aug = np.maximum(img_aug, 0.0)
        img_aug = np.minimum(img_aug, 1.0)

        # White balance
        img_aug = (img_aug - min_balance) / (max_balance - min_balance)
        img_aug = np.clip(img_aug, 0.0, 1.0)

        # return image
        imgs[i] = img_aug.astype(np.float32)


class ImgClipDataset(BaseDataset):
    """Load image from folders and split to sub-sections."""
    def __init__(self, dataset_tag, data_folder, clip_len, frm_first=0,
                 frm_step=1, clip_jump=0, blur=False, aug_flag=False):
        super(ImgClipDataset, self).__init__(dataset_tag)

        # Load config.ini for parameters
        config = ConfigParser()
        config.read(str(data_folder / 'config.ini'), encoding='utf-8')

        self.data_folder = data_folder
        self.seq_folders = sorted(list(data_folder.glob('scene_*')))
        self.reverse_flag = int(config['Data']['reverse'])
        self.start_frm = None
        if self.reverse_flag:
            self.start_frm = plb.str2tuple(config['Data']['start_frm'], int)
        self.total_sequence = int(config['Data']['total_sequence'])
        self.frm_len = plb.str2tuple(config['Data']['frm_len'], int)
        if len(self.frm_len) == 1:
            self.frm_len = self.frm_len * len(self.seq_folders)
        self.imsize = plb.str2tuple(config['Data']['img_size'], int)
        self.imsize = (self.imsize[1], self.imsize[0])  # [hei, wid]

        self.clip_len = clip_len
        self.frm_step = frm_step
        self.apply_blur = blur
        self.sigma = 1.0
        self.rng = np.random.RandomState(seed=42)
        self.data_aug = aug_flag

        pattern_num = len(list((data_folder / 'pat').glob('pat_*.png')))
        # pattern_path = list((data_folder / 'pat').glob('pat_*.png'))[-1]  Bug. This may result in pat_9.png
        pattern_path = data_folder / 'pat' / f'pat_{pattern_num - 1}.png'

        self.pattern = plb.imload(pattern_path, scale=255.0, bias=0.0, flag_tensor=False)
        if self.apply_blur:
            self.pattern = cv2.GaussianBlur(self.pattern, ksize=(7, 7), sigmaX=self.sigma)
        self.pattern = plb.a2t(self.pattern)

        if (data_folder / 'pat' / 'info.pt').exists():
            dict_load = torch.load(data_folder / 'pat' / 'info.pt')
            self.pat_info = {x: dict_load[x].unsqueeze(0) for x in dict_load}

        frm_jump = frm_step * clip_len + clip_jump
        self.samples = []
        for i, seq_folder in enumerate(self.seq_folders):
            if i >= 2 ** 11:
                break
            for frm_start in range(frm_first, self.frm_len[i], frm_jump):
                if frm_start + frm_step * clip_len > self.frm_len[i]:
                    continue
                self.samples.append((seq_folder, frm_start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_folder, frm_start = self.samples[idx]
        ret = {
            'idx': torch.Tensor([idx]),
            'frm_start': torch.Tensor([frm_start]),
            'disp_flag': torch.ones(self.clip_len).to(torch.bool),
        }

        imgs = []
        for i in range(self.clip_len):
            frm_idx = frm_start + i * self.frm_step
            img = plb.imload(seq_folder / 'img' / f'img_{frm_idx}.png', scale=255, bias=0, flag_tensor=False)
            if self.apply_blur:
                img = cv2.GaussianBlur(img, (7, 7), sigmaX=self.sigma)
            # img = plb.a2t(img)
            imgs.append(img)
        if self.data_aug:
            augment_images(imgs, self.rng)
        ret['img'] = torch.cat([plb.a2t(img) for img in imgs], dim=0)

        masks = []
        for i in range(self.clip_len):
            frm_idx = frm_start + i * self.frm_step
            mask_file = seq_folder / 'mask' / f'mask_{frm_idx}.png'
            if not mask_file.exists():
                masks.append(torch.ones_like(ret['img'][:1]))
            else:
                masks.append(plb.imload(mask_file, scale=255, bias=0))
        ret['mask'] = torch.cat(masks, dim=0)

        if (seq_folder / 'disp').exists():
            disps = []
            for i in range(self.clip_len):
                frm_idx = frm_start + i * self.frm_step
                disp_file = seq_folder / 'disp' / f'disp_{frm_idx}.png'
                if not disp_file.exists():
                    ret['disp_flag'][i] = False
                    disps.append(torch.zeros_like(ret['img'][:1]))
                else:
                    disps.append(plb.imload(seq_folder / 'disp' / f'disp_{frm_idx}.png', scale=1e2, bias=0))
            ret['disp'] = torch.cat(disps, dim=0)

        if (seq_folder / 'mask_center').exists():
            masks = []
            for i in range(self.clip_len):
                frm_idx = frm_start + i * self.frm_step
                mask = plb.imload(seq_folder / 'mask_center' / f'mask_{frm_idx}.png', scale=255, bias=0)
                masks.append(mask)
            ret['center'] = torch.cat(masks, dim=0)

        return ret

    def get_size(self):
        return self.imsize

    def get_pattern(self):
        return self.pattern.clone()

    def get_pat_info(self):
        return self.pat_info.copy()
