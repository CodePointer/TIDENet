# -*- coding: utf-8 -*-
# @Description:
#   This is a custom package that contains some useful functions.
#   'pointer' is the name of myself ^_^


# - Package Imports - #
import numpy as np
import time
from collections import OrderedDict
from pathlib import Path
import torch
import cv2


# - Coding Part: Class - #
class TimeKeeper:
    def __init__(self):
        self.start_time = time.time()

    def __str__(self):
        left_time = time.time() - self.start_time
        hour = int(left_time // 3600)
        left_time -= hour * 3600
        minute = int(left_time // 60)
        left_time -= minute * 60
        second = int(left_time)
        return f'{hour:02d}h:{minute:02d}m:{second:02d}s'


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name='DEFAULT'):
        self.name = name
        self.avg, self.sum, self.count = 0, 0, 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().mean().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        avg = self.avg
        return avg

    def clear(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def __str__(self):
        return f'[{self.name}]{self.avg:.2f}-({self.count})'


class EpochMeter:
    """Iter average & epoch average is stored"""
    def __init__(self, name='DEFAULT'):
        self.name = name
        self.iter = AverageMeter(name)
        self.epoch = AverageMeter(name)

    def update(self, val, n=1):
        self.iter.update(val, n)
        self.epoch.update(val, n)

    def get_iter(self):
        return self.iter.get()

    def clear_iter(self):
        self.iter.clear()

    def get_epoch(self):
        return self.epoch.get()

    def clear_epoch(self):
        self.iter.clear()
        self.epoch.clear()

    def __str__(self):
        return f'local: {self.iter}; epoch: {self.epoch}.'


class StopWatch(object):
    def __init__(self):
        self.timings = OrderedDict()
        self.starts = {}
        self.current_name = None
        self.sync = True

    def record(self, name, sync=True):
        self.current_name = name
        self.sync = sync
        return self

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start(self.current_name)
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync:
            torch.cuda.synchronize()
        self.stop(self.current_name)

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(time.time() - self.starts[name])

    def get(self, name=None, reduce=np.sum):
        if name is not None:
            return reduce(self.timings[name])
        else:
            ret = {}
            for k in self.timings:
                ret[k] = reduce(self.timings[k])
            return ret

    def __repr__(self):
        return ', '.join([f'{k}: {v:.2f}s' for k, v in self.get().items()])

    def __str__(self):
        return ', '.join([f'{k}: {v:.2f}s' for k, v in self.get().items()])


class VisualFactory:
    def __init__(self):
        pass

    @staticmethod
    def _crop_shape(img):
        assert len(img.shape) == 2 or len(img.shape) == 3 or len(img.shape) == 4
        if len(img.shape) == 2:
            return img.unsqueeze(0)
        elif len(img.shape) == 3:
            return img
        else:
            return img[0]

    @staticmethod
    def img_visual(img_mat, mask_mat=None, max_val=None):
        img_mat = VisualFactory._crop_shape(img_mat)
        if mask_mat is not None:
            mask_mat = VisualFactory._crop_shape(mask_mat)
            img_mat *= mask_mat
        if max_val is not None:
            if torch.min(img_mat).item() < 0:
                img_mat = img_mat / (2 * max_val) + 0.5
            else:
                img_mat = img_mat / max_val
            img_mat = torch.clamp(img_mat, 0, 1.0)
        channel = img_mat.shape[0]
        if channel == 1:
            img_mat = img_mat.repeat(3, 1, 1)
        if mask_mat is not None:
            img_mat[mask_mat.repeat(3, 1, 1) == 0] = 0
        return img_mat

    @staticmethod
    def disp_visual(disp_mat, mask_mat=None, range_val=None, color_map=cv2.COLORMAP_JET):
        """Convert disp mat into color mat."""
        disp_mat = VisualFactory._crop_shape(disp_mat)
        if mask_mat is not None:
            mask_mat = VisualFactory._crop_shape(mask_mat)
            disp_mat = disp_mat * mask_mat
        if range_val is not None:
            min_val, max_val = range_val
            disp_mat = (disp_mat - min_val) / (max_val - min_val)
            disp_mat = torch.clamp(disp_mat, 0.0, 1.0)
        disp_rgb_u8 = cv2.applyColorMap((t2a(disp_mat) * 255.0).astype(np.uint8), color_map)
        disp_rgb_u8 = cv2.cvtColor(disp_rgb_u8, cv2.COLOR_BGR2RGB)
        disp_rgb = disp_rgb_u8.astype(np.float32) / 255.0
        color_mat = a2t(disp_rgb).to(disp_mat.device)
        if mask_mat is not None:
            color_mat[mask_mat.repeat(3, 1, 1) == 0] = 0
        return color_mat

    @staticmethod
    def err_visual(err_mat, mask_mat=None, max_val=None, color_map=cv2.COLORMAP_SUMMER):
        err_mat = VisualFactory._crop_shape(err_mat)
        if mask_mat is not None:
            mask_mat = VisualFactory._crop_shape(mask_mat)
            err_mat *= mask_mat
        err_mat = torch.abs(err_mat) / max_val
        err_mat = torch.clamp(err_mat, 0.0, 1.0)
        err_rgb = cv2.applyColorMap((t2a(err_mat) * 255.0).astype(np.uint8), color_map)
        err_rgb = cv2.cvtColor(err_rgb, cv2.COLOR_BGR2RGB)
        color_mat = a2t(err_rgb.astype(np.float32) / 255.0).to(err_mat.device)
        if mask_mat is not None:
            color_mat[mask_mat.repeat(3, 1, 1) == 0] = 0
        return color_mat

    @staticmethod
    def img_concat(img_list, hei_num, wid_num, transpose=False):
        """Concat all image in img_list. RowMajor."""
        h_dim, w_dim = 1, 2
        if transpose:
            h_dim, w_dim = 2, 1

        h_stack = []
        for h in range(0, hei_num):
            idx_lf = h * wid_num
            idx_rt = min((h + 1) * wid_num, len(img_list))
            # Check max_len
            img_part = img_list[idx_lf:idx_rt]
            if len(img_part) < wid_num:
                img_part += [torch.zeros_like(img_list[0])] * (wid_num - len(img_part))
            h_stack.append(torch.cat(img_part, dim=w_dim))
        return torch.cat(h_stack, dim=h_dim)


class DepthMapConverter:
    def __init__(self, depth_map, focus):
        """
        depth_map: [1, H, W], torch.Tensor or numpy.Array.
        focus: float
        """
        self.depth_map = a2t(depth_map)
        self.imsize = depth_map.shape[-2:]
        self.focus = focus

        hei, wid = self.imsize
        self.dx = wid // 2
        self.dy = hei // 2

    def to_xyz_mat(self):
        hei, wid = self.imsize
        hh = torch.arange(0, hei).view(-1, 1).repeat(1, wid).unsqueeze(0)  # [1, H, W]
        ww = torch.arange(0, wid).view(1, -1).repeat(hei, 1).unsqueeze(0)  # [1, H, W]
        xyz_mat = torch.cat([
            (ww - self.dx) / self.focus,
            (hh - self.dy) / self.focus,
            torch.ones_like(self.depth_map)
        ], dim=0) * self.depth_map
        return xyz_mat

    def to_xyz_set(self, mask=None):
        xyz_mat = self.to_xyz_mat()
        xyz_set = xyz_mat.reshape(3, -1).permute(1, 0)
        if mask is not None:
            return xyz_set[mask.reshape(-1) > 0.0, :]
        else:
            return xyz_set

    def to_mesh(self, mask=None):  # TODO: include mask
        vertices = self.to_xyz_set().unsqueeze(0)
        hei, wid = self.imsize
        # hh = torch.arange(0, hei).view(-1, 1).repeat(1, wid).unsqueeze(0)  # [1, H, W]
        # ww = torch.arange(0, wid).view(1, -1).repeat(hei, 1).unsqueeze(0)  # [1, H, W]
        faces = []
        for h in range(0, hei - 1):
            for w in range(0, wid - 1):
                lf_up = h * wid + w
                lf_dn = (h + 1) * wid + w
                rt_up = h * wid + w + 1
                rt_dn = (h + 1) * wid + w + 1
                triangles = torch.as_tensor([
                    [lf_up, rt_dn, rt_up],
                    [lf_up, lf_dn, rt_dn]
                ], dtype=torch.int)  # [2, 3]
                faces.append(triangles)
        faces = torch.cat(faces, dim=0).unsqueeze(0)
        return vertices, faces


# - Coding Part: Funcs - #
def subfolders(folder):
    if folder is None or folder == '':
        return None
    if isinstance(folder, str):
        folder = Path(folder)
    return sorted([x for x in folder.glob('*') if x.is_dir()])


def a2t(in_array, permute=True):
    """np.ndarray -> torch.Tensor.
        Output shape:
            [H, W] -> [1, H, W]
            [H, W, C] -> [C, H, W] (permute = True)
                      -> [H, W, C] (permute = False)
            other shape -> same
    """
    if isinstance(in_array, torch.Tensor):
        return in_array
    out_tensor = torch.from_numpy(in_array)
    if len(out_tensor.shape) == 2:
        return out_tensor.unsqueeze(0)
    elif len(out_tensor.shape) == 3:
        if permute:
            return out_tensor.permute(2, 0, 1)
        else:
            return out_tensor
    else:
        return out_tensor


def t2a(in_tensor, permute=True):
    """torch.Tensor -> np.ndarray.
        Output shape:
            [1, H, W] -> [H, W]
            [C, H, W] -> [H, W, C] (permute = True)
                      -> [C, H, W] (permute = False)
            other shape -> same
    """
    if isinstance(in_tensor, np.ndarray):
        return in_tensor
    out_tensor = in_tensor.detach().cpu()
    if len(out_tensor.shape) == 3:
        if out_tensor.shape[0] == 1:
            return out_tensor.squeeze(0).numpy()
        else:
            if permute:
                return out_tensor.permute(1, 2, 0).numpy()
            else:
                return out_tensor.numpy()
    else:
        return out_tensor.numpy()


def imload(path, scale=255.0, bias=0.0, flag_tensor=True):
    """Load image with default type."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Failed to read file: {path}')
    img = (img.astype(np.float32) - bias) / scale
    if flag_tensor:
        img = a2t(img)
    return img


def imsave(path, img, scale=255.0, bias=0.0, img_type=np.uint8, mkdir=False):
    """Save image."""
    img_copy = t2a(img).copy()
    img_copy = img_copy.astype(np.float32) * scale + bias
    # Check folder
    if mkdir:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_copy.astype(img_type))


def imviz(img, name='DEFAULT', wait=0, normalize=None):
    """Visualize image. Accept normalization function for visualize."""
    img_copy = t2a(img).copy()
    if img_copy.dtype == np.uint8:
        img_copy = img_copy.astype(np.float32) / 255.0
    if isinstance(normalize, list):
        min_val, max_val = 0, 255
        if len(normalize) == 0:
            min_val, max_val = np.min(img_copy), np.max(img_copy)
        elif len(normalize) == 2:
            min_val, max_val = normalize
        else:
            raise ValueError(f"Normalize length is not valid: {len(normalize)}")
        img_copy = (img_copy - min_val) / (max_val - min_val)
        img_copy = np.clip(img_copy, 0, 1.0)
    cv2.imshow(name, (img_copy * 255.0).astype(np.uint8))
    return cv2.waitKey(wait)


def str2tuple(input_string, item_type=float):
    """.ini file processing function."""
    return tuple([item_type(x.strip()) for x in input_string.split(',')])
