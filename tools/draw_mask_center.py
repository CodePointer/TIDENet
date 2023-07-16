# -*- coding: utf-8 -*-

# @Time:      2022/10/4 19:40
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      draw_mask_center.py
# @Software:  PyCharm
# @Description:
#   Run this description with different main_folder path. It will draw all the mask_center mask.

# - Package Imports - #
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np


# - Coding Part - #
class MaskCenterDrawer:
    def __init__(self):
        self.params = cv2.SimpleBlobDetector_Params()

        self.params.minThreshold = 10
        self.params.maxThreshold = 256
        self.params.thresholdStep = 2
        self.params.minDistBetweenBlobs = 0
        self.params.minRepeatability = 5

        self.params.filterByColor = True
        self.params.blobColor = 255

        self.params.filterByArea = True
        self.params.minArea = 0
        self.params.maxArea = 50

        self.params.filterByCircularity = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False

        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def detect(self, img_u8):
        key_points = self.detector.detect(img_u8)
        dots = [x.pt for x in key_points]
        return dots

    def draw_center(self, img_u8, dots):
        mask_u8 = np.zeros_like(img_u8)
        for dot in dots:
            x, y = dot
            h, w = round(y), round(x)
            mask_u8[h, w] = 255
        return mask_u8


def img2center(img_file, center_file, drawer=None):
    if drawer is None:
        drawer = MaskCenterDrawer()
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    dots = drawer.detect(img)
    mask = drawer.draw_center(img, dots)
    cv2.imwrite(str(center_file), mask)


def main(data_folder):
    folder_list = [x for x in data_folder.glob('scene_*') if x.is_dir()]
    drawer = MaskCenterDrawer()
    for folder in folder_list:
        (folder / 'mask_center').mkdir(exist_ok=True)
        img_folder = folder / 'img'
        total_frm = len(list(img_folder.glob('*.png')))
        for frm_idx in tqdm(range(total_frm), desc=folder.name):
            img2center(
                img_file=img_folder / f'img_{frm_idx}.png',
                center_file=folder / 'mask_center' / f'mask_{frm_idx}.png',
                drawer=drawer
            )


if __name__ == '__main__':
    main(Path('./data/3_Non-rigid-Real'))
