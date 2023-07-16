# -*- coding: utf-8 -*-

# @Time:      2022/12/15 23:06
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      mpf_visualizer.py
# @Software:  PyCharm
# @Description:
#   This file is for debugging. Visualize the multi-frame pattern flow.

# - Package Imports - #
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# import pointerlib as plb
import utils.pointerlib as plb
from utils.pattern_flow import MPFlowEstimator
from models.supervise import PFDistLoss


# - Coding Part - #
class MPFVisualizer:
    def __init__(self, clip_len=8, color_map=cv2.COLORMAP_JET):
        tmp_color_list = np.arange(0, 255).astype(np.uint8)
        color_list = cv2.applyColorMap(tmp_color_list, color_map)
        self.color_list = [x.reshape(-1).tolist() for x in color_list]
        self.clip_len = clip_len
        self.mpf_estimator = MPFlowEstimator(clip_len)

    def get_color(self, i):
        if isinstance(i, int):
            idx = np.mod(i * 13, 255)
        elif isinstance(i, float):
            idx = int(i * 254.0)
        else:
            return None
        return self.color_list[idx]

    @staticmethod
    def to_pt(pair, reverse=False):
        if reverse:
            return int(pair[1]), int(pair[0])
        else:
            return int(pair[0]), int(pair[1])

    def run(self, mask_centers, imgs):
        mpf_distribution = None
        total_frm = len(mask_centers)
        res = []

        for frm_start in tqdm(range(0, total_frm, self.clip_len), desc='Computing'):
            if frm_start + self.clip_len > total_frm:
                break

            mpf_dots, mpf_distribution = self.mpf_estimator.run(
                mask_centers[frm_start:frm_start + self.clip_len],
                mpf_distribution
            )  # mpf_dots: [T, 2, Kc]

            dot_num = mpf_dots.shape[-1]
            line_canvas = np.zeros_like(imgs[0])
            for t in range(self.clip_len):
                dot_canvas = np.zeros_like(line_canvas)
                for i in range(1, dot_num):
                    if mpf_dots[t, 0, i] < 0:
                        continue

                    # Line
                    if t > 0 and mpf_dots[t - 1, 0, i] >= 0:
                        cv2.line(line_canvas,
                                 self.to_pt(mpf_dots[t - 1, :, i]),
                                 self.to_pt(mpf_dots[t, :, i]),
                                 self.get_color(i), 1)

                    # Dot for current image
                    cv2.circle(dot_canvas,
                               self.to_pt(mpf_dots[t, :, i]),
                               2,
                               self.get_color(i), -1)
                mpf_canvas = cv2.add(line_canvas, dot_canvas)
                res_img = cv2.addWeighted(imgs[frm_start + t], 0.3, mpf_canvas, 0.7, 0)
                res.append(res_img)

        return res


def main():
    seq_folder = Path('./data/3_Non-rigid-Real/scene_0000')
    total_frm = 32  # 2000
    clip_len = 8

    img_set = []  # np.uint8, 3
    mask_set = []  # torch.float32
    for frm_idx in tqdm(range(total_frm), desc='Loading'):
        img = plb.imload(seq_folder / 'img' / f'img_{frm_idx}.png', flag_tensor=False)
        img_u8 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_set.append(img_u8)

        mask = plb.imload(seq_folder / 'mask_center' / f'mask_{frm_idx}.png')
        mask_set.append(mask)

    visualizer = MPFVisualizer(clip_len)
    res_set = visualizer.run(mask_set, img_set)

    # Show image
    i = 0
    while True:
        key = plb.imviz(res_set[i], 'img', wait=200)
        if key == 27:
            break
        i = (i + 1) % total_frm

    pass


if __name__ == '__main__':
    main()
