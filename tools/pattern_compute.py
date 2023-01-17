# -*- coding: utf-8 -*-

import utils.pointerlib as plb
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List


class PatBlob:
    def __init__(self, x, y, idx):
        self._idx = idx
        self._x = x
        self._y = y
        self.rgb = np.array([255, 255, 255], dtype=np.uint8)

        self.edges = {}
        self.unconnected_edge_num = 0

        self.cam_pos = []
        self.acc_thred = 3.0

        self.candidates = []

    @property
    def pat_pos(self):
        return round(self._x), round(self._y)

    @property
    def idx(self):
        return self._idx

    @staticmethod
    def create_from_field(center_pattern, rect_pattern_field):
        cen_pattern = plb.imload(center_pattern, flag_tensor=False)
        pat_field = plb.imload(rect_pattern_field, scale=1.0, flag_tensor=False).astype(np.uint16)
        rgb_pattern = plb.imload('rect_pattern_field_vis.png', scale=1.0, flag_tensor=False).astype(np.uint8)

        hei, wid = cen_pattern.shape
        blob_num = int(pat_field.max())
        pat_blob_list = [None for i in range(blob_num + 1)]

        # Create blob
        for h in range(hei):
            for w in range(wid):
                if cen_pattern[h, w] == 1.0:
                    idx = pat_field[h, w].item()
                    pat_blob_list[idx] = PatBlob(w, h, idx)
                    pat_blob_list[idx].rgb = rgb_pattern[h, w]

        # Set edge
        for h in range(hei - 1):
            for w in range(wid - 1):
                idx_i = pat_field[h, w]
                for dh, dw in [(0, 1), (1, 0)]:
                    idx_j = pat_field[h + dh, w + dw]
                    if idx_i != idx_j and min(idx_i, idx_j) > 0:
                        pat_blob_list[idx_i].add_edge(idx_j)
                        pat_blob_list[idx_j].add_edge(idx_i)

        return pat_blob_list

    def add_edge(self, idx_nbr):
        self.edges[idx_nbr] = False

    def set_cam_pos(self, frm_idx, cam_pos):
        self.cam_pos.append((frm_idx, cam_pos))

    def get_cam_coord(self, frm_idx):
        assert self.available(frm_idx)
        x, y = self.cam_pos[-1][1]
        return round(x), round(y)

    def available(self, frm_now):
        if len(self.cam_pos) == 0:
            return False
        frm_num, cam_pos = self.cam_pos[-1]
        return frm_num == frm_now

    @staticmethod
    def update_edge_status(pat_blob_list, idx_set: Optional[List] = None):
        if idx_set is None:
            idx_set = range(1, len(pat_blob_list))
        for idx in idx_set:
            pat_blob = pat_blob_list[idx]
            if not pat_blob.available():
                continue
            for nbr_idx in pat_blob.edge_current:
                status = pat_blob.available() and pat_blob_list[nbr_idx].available()
                pat_blob.set_edge_status(nbr_idx, status)


class MaskDrawer:
    def __init__(self, img_size=None):
        if img_size is None:
            img_size = [480, 848]
        self.params = cv2.SimpleBlobDetector_Params()

        self.params.minThreshold = 10
        self.params.maxThreshold = 256
        self.params.thresholdStep = 3
        self.params.minDistBetweenBlobs = 0
        self.params.minRepeatability = 3

        self.params.filterByColor = True
        self.params.blobColor = 255

        self.params.filterByArea = True
        self.params.minArea = 0
        self.params.maxArea = 20

        self.params.filterByCircularity = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False

        self.detector = cv2.SimpleBlobDetector_create(self.params)

        self.imsize = tuple(img_size)

        self.adj_matrix = None
        self.field_map = None

    def set_imsize(self, hei, wid):
        self.imsize = (hei, wid)

    def detect_dots(self, pat):
        pat_u8 = plb.t2a(pat * 255).astype(np.uint8)
        key_points = self.detector.detect(pat_u8)
        dots = []
        for key_point in key_points:
            x, y = key_point.pt
            dots.append((x, y))
        return dots

    def is_inside(self, h, w):
        hei, wid = self.imsize
        return 0 <= h < hei and 0 <= w < wid

    def create_field(self, dots, mask=None, max_dxy=None):
        delta = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        fields = np.zeros(self.imsize, np.uint16)
        dot_max_inf_num = len(dots) + 1024
        if mask is None:
            fields[:, 700:] = dot_max_inf_num
        else:
            mask = plb.t2a(mask)
            fields[mask < 1.0] = dot_max_inf_num

        pbar = tqdm(total=(mask > 0).sum())
        dot_queue = []
        for i, dot in enumerate(dots):
            x, y = dot
            w, h = round(x), round(y)
            fields[h, w] = i + 1
            dot_queue.append((w, h, i + 1))
            pbar.update(1)

        while len(dot_queue) > 0:
            w, h, idx = dot_queue[0]
            dot_queue.pop(0)
            for dh, dw in delta:
                hn, wn = h + dh, w + dw
                if not self.is_inside(hn, wn):
                    continue

                if max_dxy is not None:
                    max_dx, max_dy = max_dxy
                    w_src, h_src = [round(x) for x in dots[idx - 1]]
                    if abs(w_src - wn) > max_dx or abs(h_src - hn) > max_dy:
                        continue

                if fields[hn, wn] == 0:
                    fields[hn, wn] = idx
                    dot_queue.append((wn, hn, idx))
                    pbar.update(1)

        fields[fields == dot_max_inf_num] = 0
        self.field_map = fields

        if max_dxy is not None:
            return fields

        hei, wid = self.imsize
        dot_num = len(dots)
        self.adj_matrix = [set() for x in range(dot_num + 1)]
        for h in range(0, hei - 1):
            for w in range(0, wid - 1):
                idx_i = self.field_map[h, w]
                for dh, dw in [(0, 1), (1, 0)]:
                    hn = h + dh
                    wn = w + dw
                    idx_j = self.field_map[hn, wn]
                    if idx_i != idx_j and min(idx_i, idx_j) > 0:
                        self.adj_matrix[idx_i].add(idx_j)
                        self.adj_matrix[idx_j].add(idx_i)

        return fields

    def get_adj_edge(self):
        tmp_list = []
        max_adj = max([len(x) for x in self.adj_matrix])
        for adj_row in self.adj_matrix:
            tmp_row = np.array(sorted(adj_row))
            np_row = np.zeros(max_adj, dtype=np.int64)
            np_row[:len(tmp_row)] = tmp_row
            tmp_list.append(np_row)
        adj_mat = np.stack(tmp_list, axis=0)
        return adj_mat

    def draw_mask(self, dots):
        mask_u8 = np.zeros(self.imsize, np.uint8)
        for dot in dots:
            x, y = dot
            w, h = round(x), round(y)
            mask_u8 = cv2.circle(mask_u8, (w, h), 1, color=255, thickness=-1)

        adj_len = 0 if self.adj_matrix is None else len(self.adj_matrix)
        for i in range(1, adj_len):
            x, y = dots[i - 1]
            w, h = round(x), round(y)
            for idx_n in self.adj_matrix[i]:
                xn, yn = dots[idx_n - 1]
                wn, hn = round(xn), round(yn)
                mask_u8 = cv2.line(mask_u8, (w, h), (wn, hn), color=128)

        mask = plb.a2t(mask_u8.astype(np.float32) / 255.0)
        return mask

    def draw_coord(self, dots, field):
        hei, wid = self.imsize
        xcoord_mat = np.zeros_like(field)
        for h in range(hei):
            for w in range(wid):
                idx = field[h, w]
                if idx == 0:
                    continue
                w_cen, h_cen = [round(x) for x in dots[idx - 1]]
                xcoord_mat[h, w] = w_cen
        return xcoord_mat


class IdGenerator:
    def __init__(self, start_id=42):
        self.next_id = start_id + 1

    def get_new(self):
        next_id = self.next_id
        self.next_id += 1
        return next_id


def create_pattern_field(pattern_name, pattern_mask):
    pattern = plb.imload(pattern_name)
    mask = plb.imload(pattern_mask)

    # Write pid & xp
    pid_drawer = MaskDrawer(img_size=pattern.shape[-2:])
    dots = pid_drawer.detect_dots(pattern)
    pat_pid = pid_drawer.create_field(dots, mask, max_dxy=[800, 4])
    pat_xp = pid_drawer.draw_coord(dots, pat_pid)
    # mask_viz = pid_drawer.draw_mask(dots)
    # plb.imviz(mask_viz, 'mask', 10)
    # plb.imviz(pattern, 'pat', 0)

    # Write neighbor
    edge_drawer = MaskDrawer(img_size=pattern.shape[-2:])
    edge_drawer.create_field(dots, mask)
    pos = np.array(dots).astype(np.int64)
    pos = np.concatenate([np.zeros([1, 2], dtype=pos.dtype), pos], axis=0)
    edge_mat = edge_drawer.get_adj_edge()

    # Write diff
    blob_num = pos.shape[0]
    diff_mat = np.zeros([*edge_mat.shape, 2], dtype=np.float32)
    for i in range(1, blob_num):
        for j in range(len(edge_mat[i])):
            if edge_mat[i, j] == 0:
                continue
            i_nbr = edge_mat[i, j]
            diff_mat[i, j, :] = pos[i] - pos[i_nbr]

    # Save
    save_dict = {
        'pid': plb.a2t(pat_pid.astype(np.int64)),  # [1, H, W]
        'xp': plb.a2t(pat_xp.astype(np.float32)),  # [1, H, W]
        'pos': plb.a2t(pos.astype(np.int64)),  # [1, Kp, 2]
        'edge': plb.a2t(edge_mat.astype(np.int64)),  # [1, Kp, 8]
        'diff': plb.a2t(diff_mat.astype(np.float32))  # [2, Kp, 8]
    }
    torch.save(save_dict, str(pattern_name.parent / 'pat_info.pt'))
    return


if __name__ == '__main__':
    create_pattern_field(
        pattern_name=Path('C:/SLDataSet/TADE/pat_0.png'),
        pattern_mask=Path('C:/SLDataSet/TADE/mask.png')
    )
    pass
