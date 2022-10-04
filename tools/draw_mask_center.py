# -*- coding: utf-8 -*-

# @Time:      2022/10/4 19:40
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      draw_mask_center.py.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from tqdm import tqdm
from pathlib import Path

from tools.pattern_compute import MaskDrawer
import utils.pointerlib as plb


# - Coding Part - #
def draw_center(data_folder):
    folder_list = plb.subfolders(data_folder)
    drawer = MaskDrawer(img_size=[480, 848])
    for folder in folder_list:
        (folder / 'mask_center').mkdir(exist_ok=True)
        img_folder = folder / 'img'
        total_frm = len(list(img_folder.glob('*.png')))
        for frm_idx in tqdm(range(total_frm), desc=folder.name):
            img = plb.imload(img_folder / f'img_{frm_idx}.png', 255.0)
            dots = drawer.detect_dots(img)
            mask = drawer.draw_mask(dots)
            plb.imsave(folder / 'mask_center' / f'mask_{frm_idx}.png', mask, 255.0, mkdir=True)


if __name__ == '__main__':
    draw_center(Path('C:/SLDataSet/Data0605'))
