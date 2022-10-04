# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import cv2
import numpy as np
import utils.pointerlib as plb
from models.base_dataset import BaseDataset


# - Coding Part - #
def augment_image(img, rng, max_blur=1.5, max_noise=10.0, max_sp_noise=0.001):
    # get min/max values of image
    min_val = np.min(img)
    max_val = np.max(img)

    # init augmented image
    img_aug = img

    # gaussian smoothing
    if rng.uniform(0, 1) < 0.5:
        img_aug = cv2.GaussianBlur(img_aug, (5, 5), rng.uniform(0.2, max_blur))

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

    # return image
    return img_aug.astype(np.float32)


class ImgClipDataset(BaseDataset):
    """Load image from folders and split to sub-sections."""
    def __init__(self, dataset_tag, seq_folders, clip_len, pattern_path, 
                 frm_step=1, clip_jump=0, blur=False, aug_flag=False):
        super(ImgClipDataset, self).__init__(dataset_tag)
        self.seq_folders = seq_folders
        self.clip_len = clip_len
        self.frm_step = frm_step
        self.apply_blur = blur
        self.sigma = 1.0
        self.rng = np.random.RandomState(seed=42)
        self.data_aug = aug_flag

        self.pattern = plb.imload(pattern_path, scale=255.0, bias=0.0, flag_tensor=False)
        if self.apply_blur:
            self.pattern = cv2.GaussianBlur(self.pattern, ksize=(7, 7), sigmaX=self.sigma)
        self.pattern = plb.a2t(self.pattern)

        if (pattern_path.parent / 'pat_info.pt').exists():
            dict_load = torch.load(pattern_path.parent / 'pat_info.pt')
            self.pat_info = {x: dict_load[x].unsqueeze(0) for x in dict_load}

        self.imsize = self.pattern.shape[-2:]

        frm_jump = frm_step * clip_len + clip_jump
        self.samples = []
        for seq_folder in seq_folders:
            total_frm = len(list((seq_folder / 'img').glob('*.png')))
            for frm_start in range(0, total_frm, frm_jump):
                if frm_start + frm_step * clip_len > total_frm:
                    continue
                self.samples.append((seq_folder, frm_start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_folder, frm_start = self.samples[idx]
        ret = {'idx': torch.Tensor([idx]), 'frm_start': torch.Tensor([frm_start])}

        imgs = []
        for i in range(self.clip_len):
            frm_idx = frm_start + i * self.frm_step
            img = plb.imload(seq_folder / 'img' / f'img_{frm_idx}.png', scale=255, bias=0, flag_tensor=False)
            if self.apply_blur:
                img = cv2.GaussianBlur(img, (7, 7), sigmaX=self.sigma)
            if self.data_aug:
                img = augment_image(img, self.rng)
            img = plb.a2t(img)
            imgs.append(img)
        ret['img'] = torch.cat(imgs, dim=0)

        if (seq_folder / 'disp').exists():
            disps = []
            for i in range(self.clip_len):
                frm_idx = frm_start + i * self.frm_step
                disp = plb.imload(seq_folder / 'disp' / f'disp_{frm_idx}.png', scale=1e2, bias=0)
                disp[:, :, :320] = 0.0
                disps.append(disp)
            ret['disp'] = torch.cat(disps, dim=0)

        if (seq_folder / 'mask_obj').exists():
            masks = []
            for i in range(self.clip_len):
                frm_idx = frm_start + i * self.frm_step
                mask = plb.imload(seq_folder / 'mask_obj' / f'mask_{frm_idx}.png', scale=255, bias=0)
                masks.append(mask)
            ret['mask'] = torch.cat(masks, dim=0)
        else:
            ret['mask'] = torch.ones_like(ret['img'])

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
