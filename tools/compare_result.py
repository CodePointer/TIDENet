# -*- coding: utf-8 -*-

# @Time:      2022/9/20 18:56
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      compare_result.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import shutil
import configparser
import cv2
import numpy as np
import pointerlib as plb
from tqdm import tqdm

from utils.args import get_args


# - Coding Part - #
class Evaluator:
    """Estimated disparity comparison."""

    def __init__(self, data_dir, res_dir, rep_dir, mask_name, params=None, silence=False, rgb_save=False):
        self.data_dir = data_dir
        self.res_dir = res_dir
        self.rep_dir = rep_dir
        self.rep_dir.mkdir(exist_ok=True, parents=True)
        self.mask_name = mask_name
        self.rgb_save = rgb_save

        self.silence = silence
        # Get seq_set & roi set
        if (data_dir / 'roi.ini').exists():
            roi_config = configparser.ConfigParser()
            roi_config.read(str(data_dir / 'roi.ini'))
            self.seq_rois = dict()
            for seq_name in roi_config['ROI'].keys():
                self.seq_rois[seq_name] = plb.str2tuple(roi_config['ROI'][seq_name], item_type=int)
        else:
            self.seq_rois = dict()
            for seq_folder in plb.subfolders(self.data_dir):
                self.seq_rois[seq_folder.name] = None

        # Copy ini
        if params is not None:
            shutil.copy(str(params), str(self.rep_dir / params.name))

        self.gif_interval = 2

    def my_range(self, start, end, step=1, desc=''):
        if self.silence:
            return range(start, end, step)
        else:
            return tqdm(range(start, end, step), desc=desc)

    def iter_wrapper(self, iterator, desc=''):
        if self.silence:
            return iterator
        else:
            return tqdm(iterator, desc=desc)

    def save_csv_summary(self, metric_list, err_total):
        # Save summary errs for every metric
        for metric in metric_list:
            frm_num = len(err_total[metric][0])
            seq_num = len(err_total[metric])
            metric_file = self.rep_dir / f'err_{metric}.csv'
            with open(str(metric_file), 'w') as file:
                for frm_idx in range(frm_num):
                    for seq_idx in range(seq_num):
                        if frm_idx == 0:
                            file.write(f'{err_total[metric][seq_idx][frm_idx]},')
                        else:
                            if metric in ['10', '20', '50']:
                                file.write(f'{err_total[metric][seq_idx][frm_idx]:.3f},')
                            else:
                                file.write(f'{err_total[metric][seq_idx][frm_idx]:.3f},')
                    file.write('\n')

        # Save to summary
        def avg_cal(vals):
            avg_res = 0.0
            for val in vals:
                avg_res += val / len(vals)
            return avg_res

        summary_file = self.rep_dir / f'err_summary.csv'
        with open(str(summary_file), 'w') as file:
            # Write seq_title
            seq_name = [x[0] for x in err_total[metric_list[0]]]
            line_str = ','.join([''] + seq_name) + '\n'
            file.write(line_str)
            # Write value
            seq_num = len(err_total[metric_list[0]])
            for metric in metric_list:
                file.write(metric + ',')
                for seq_idx in range(seq_num):
                    avg_err = avg_cal(err_total[metric][seq_idx][1:])
                    file.write(f'{avg_err:.3f},')
                file.write('\n')
        pass

    def compare_disparity(self):
        # Set error metrics
        metric_list = ['10', '20', '50', 'avg', 'mse']
        err_total = {k: [] for k in metric_list}

        # Check all seq_folders
        for seq_name in self.seq_rois.keys():

            gt_folder = self.data_dir / seq_name / 'disp'
            est_folder = self.res_dir / seq_name / 'disp'
            mask_folder = self.data_dir / seq_name / self.mask_name

            # Create err list for output
            err_seq = {k: [seq_name, ] for k in metric_list}

            # if rgb_save:
            #     dp_folder = self.ret_folder / seq_name / 'disp'
            #     avg_folder = self.ret_folder / seq_name / 'avg_err'
            #     step_folder = self.ret_folder / seq_name / 'step_err'
            #     dp_folder.mkdir(exist_ok=True, parents=True)
            #     avg_folder.mkdir(exist_ok=True, parents=True)
            #     step_folder.mkdir(exist_ok=True, parents=True)

            # Process every frame in sequence
            total_num = len(list(est_folder.glob('*.png')))
            for frm_idx in self.my_range(1, total_num, desc=seq_name):
                # Load
                disp_est = plb.imload(est_folder / f'disp_{frm_idx}.png', scale=1e2, flag_tensor=False)
                if self.rgb_save:
                    # disparity
                    disp_vis = plb.t2a(plb.VisualFactory.disp_visual(plb.a2t(disp_est), range_val=[50.0, 300.0]))  # 50
                    dp_folder = self.rep_dir / seq_name / 'disp'
                    plb.imsave(dp_folder / f'disp_{frm_idx}.png', disp_vis, mkdir=True)
                    # plb.imviz(disp_vis, 'disp', 10)

                if not gt_folder.exists():
                    disp_gt = None
                else:
                    disp_gt = plb.imload(gt_folder / f'disp_{frm_idx}.png', scale=1e2, flag_tensor=False)
                if not (mask_folder / f'mask_{frm_idx}.png').exists():
                    mask_gt = np.ones_like(disp_gt).astype(np.bool)
                    mask_gt[:, :208] = False
                else:
                    mask_gt = plb.imload(mask_folder / f'mask_{frm_idx}.png', flag_tensor=False).astype(np.bool)
                total_pix = np.sum(mask_gt.astype(np.float32))
                if total_pix == 0:
                    total_pix = 1

                # Calculate err
                if disp_gt is not None:
                    err10_mat = (np.abs(disp_gt - disp_est) > 1.0).astype(np.float32)
                    err_seq['10'].append(np.sum(err10_mat[mask_gt]) / total_pix * 100.0)
                    err20_mat = (np.abs(disp_gt - disp_est) > 2.0).astype(np.float32)
                    err_seq['20'].append(np.sum(err20_mat[mask_gt]) / total_pix * 100.0)
                    err50_mat = (np.abs(disp_gt - disp_est) > 5.0).astype(np.float32)
                    err_seq['50'].append(np.sum(err50_mat[mask_gt]) / total_pix * 100.0)
                    avg_mat = np.abs(disp_gt - disp_est)
                    err_seq['avg'].append(np.sum(avg_mat[mask_gt]) / total_pix)
                    mse_mat = avg_mat ** 2
                    err_seq['mse'].append(np.sum(mse_mat[mask_gt]) / total_pix)

                    if self.rgb_save:
                        # Error image save
                        avg_vis = plb.t2a(plb.VisualFactory.err_visual(plb.a2t(avg_mat), mask_mat=plb.a2t(mask_gt),
                                                                       max_val=10.0, color_map=cv2.COLORMAP_SUMMER))
                        avg_vis = cv2.cvtColor(avg_vis, cv2.COLOR_BGR2RGB)
                        avg_folder = self.rep_dir / seq_name / 'avg_err'
                        plb.imsave(avg_folder / f'err_{frm_idx}.png', avg_vis, mkdir=True)

                        # Extract step image
                        step_err = np.zeros_like(avg_mat)
                        step_err[(1.0 - err10_mat).astype(np.bool)] = 1.0
                        step_err[((1.0 - err20_mat) * err10_mat).astype(np.bool)] = 2.0
                        step_err[((1.0 - err50_mat) * err20_mat).astype(np.bool)] = 3.0
                        step_err[err50_mat.astype(np.bool)] = 4.0

                        # Create BAR
                        # step_err[:120, :100] = 1.0
                        # step_err[120:240, :100] = 2.0
                        # step_err[240:360, :100] = 3.0
                        # step_err[360:, :100] = 4.0
                        # mask_gt[:, :100] = True

                        step_vis = plb.t2a(plb.VisualFactory.err_visual(plb.a2t(step_err), mask_mat=plb.a2t(mask_gt),
                                                                        max_val=4.0, color_map=cv2.COLORMAP_WINTER))
                        step_vis = cv2.cvtColor(step_vis, cv2.COLOR_BGR2RGB)
                        step_folder = self.rep_dir / seq_name / 'step_err'
                        plb.imsave(step_folder / f'err_{frm_idx}.png', step_vis)

            # Append err_seq to total
            for metric in metric_list:
                err_total[metric].append(err_seq[metric])

        self.save_csv_summary(metric_list, err_total)

    def draw_gt(self):
        for seq_name in self.seq_rois.keys():
            # Set roi for visualization
            roi = self.seq_rois[seq_name]
            roi = (208, 848, 0, 480)

            if seq_name != '20210926_204046':
                continue

            img_folder = self.data_dir / seq_name / 'img'
            img_gif = img_folder.parent / 'img.gif'
            plb.GifWriter(img_gif, roi, fps=5).run_folder(img_folder, interval=self.gif_interval)
            continue

            gt_folder = self.data_dir / seq_name / 'disp_rgb'
            if not gt_folder.exists():
                disp_folder = self.data_dir / seq_name / 'disp'
                total_frm = len(list(disp_folder.glob('*.png')))
                for frm_idx in range(total_frm):
                    disp_gt = plb.imload(disp_folder / f'disp_{frm_idx}.png', scale=1e2)
                    disp_vis = plb.VisualFactory.disp_visual(disp_gt, range_val=[50.0, 300.0])
                    plb.imsave(gt_folder / f'disp_{frm_idx}.png', disp_vis, mkdir=True)

            gt_gif = gt_folder.parent / 'disp_rgb.gif'
            # plb.GifWriter(gt_gif, roi).run_folder(gt_folder, interval=self.gif_interval, max_frm=256)

    def draw_gif(self, interval=1):
        for seq_name in self.seq_rois.keys():
            # Set roi for visualization
            roi = self.seq_rois[seq_name]
            if roi is None:
                roi = (208, 848, 0, 480)

            rgb_sets = plb.subfolders(self.rep_dir / seq_name)
            for rgb_set in self.iter_wrapper(rgb_sets, desc=f'Gif:{seq_name}'):
                gif_name = rgb_set.parent / f'{rgb_set.name}.gif'
                gif_folder = rgb_set
                plb.GifWriter(gif_name, roi, fps=5).run_folder(gif_folder, interval=interval)


def evaluate_res(args):
    epoch_num = 200
    mask_name = 'mask_obj'

    data_folder = Path(args.train_dir)
    exp_tag = '-'.join([args.argset, args.run_tag])

    res_dir = Path(
        data_folder.parent,
        f'{data_folder.name}-out',
        exp_tag,
        'output',
        data_folder.name,
        f'epoch_{epoch_num:05}',
    )
    params = Path(
        data_folder.parent,
        f'{data_folder.name}-out',
        exp_tag,
        'params.ini',
    )
    rep_dir = Path(
        data_folder.parent,
        f'{data_folder.name}-rep',
        exp_tag,
    )
    evaluator = Evaluator(
        data_dir=data_folder,
        res_dir=res_dir,
        rep_dir=rep_dir,
        params=params,
        mask_name=mask_name,
        rgb_save=True
    )
    evaluator.compare_disparity()
    evaluator.draw_gif()


def main():
    args = get_args()
    for x in ['online4']:
        args.run_tag = x
        evaluate_res(args)


if __name__ == '__main__':
    main()
