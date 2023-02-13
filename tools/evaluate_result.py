# -*- coding: utf-8 -*-

# @Time:      2023/01/22 15:40
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      evaluate_result.py
# @Software:  PyCharm
# @Description:
#   This file is used for disparity map evaluation.
#   Notice the coordinate for open3d is: left x, up y, left handed.

# - Package Imports - #
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import cv2
import torch
import openpyxl
from tqdm import tqdm
import re
import shutil
# import open3d as o3d

import pointerlib as plb


# - Coding Part - #
class Evaluator:
    def __init__(self, workbook, flush_flag):
        self.workbook = openpyxl.load_workbook(str(workbook))
        self.workbook_path = workbook
        self.flush_flag = flush_flag

    # def run(self, sheet_names=None):
    #     # Calculate err
    #     # if sheet_names is None:
    #     #     sheet_names = self.workbook.get_sheet_names()
    #     for scene_name in sheet_names:
    #         self.process_dataset(scene_name)

    def process_dataset(self, scene_name, mask_flag=False):
        work_sheet = self.workbook[scene_name]
        data_path = Path(work_sheet['B1'].value)
        out_path = Path(work_sheet['B2'].value)

        # load parameters
        config = ConfigParser()
        config.read(str(data_path / 'config.ini'), encoding='utf-8')
        seq_num = int(config['Data']['total_sequence'])
        frm_len = plb.str2tuple(config['Data']['frm_len'], int)
        if len(frm_len) == 1:
            frm_len = frm_len * seq_num
        wid, hei = plb.str2tuple(config['Data']['img_size'], item_type=int)

        # Get all exp_tags
        start_row = 5
        exp_set = sorted([x for x in out_path.glob('*') if x.is_dir()])
        cmp_sets = []
        exp_num = len(exp_set)
        for i, exp_path in enumerate(exp_set):
            work_sheet.cell(row=start_row + i, column=1).value = exp_path.name
            sub_folders = [x for x in (exp_path / 'output' / data_path.name).glob('*') if x.is_dir()]
            res_path = sorted(sub_folders)[-1]
            work_sheet.cell(row=start_row + i, column=2).value = str(res_path)
            cmp_sets.append(self._build_up_cmp_set(data_path, res_path, frm_len, mask_flag))
            
        # Evaluate: column = 3,4,5,6
        prog_bar = tqdm(total=sum([len(x) for x in cmp_sets]))
        for i in range(exp_num):
            prog_bar.set_description(f'{scene_name}[{exp_set[i].name}]')
            row_idx = start_row + i

            # Check if skip or not
            flag_all_filled = True
            for col_idx in [3, 4, 5, 6]:
                cell_res = work_sheet.cell(row=row_idx, column=col_idx)
                flag_all_filled = flag_all_filled and cell_res.value is not None
            if not self.flush_flag and flag_all_filled:
                prog_bar.update(len(cmp_sets[i]))
                continue

            # Evaluate result and fill results
            res = self._evaluate_exp_outs(cmp_sets[i], prog_bar)
            work_sheet.cell(row=row_idx, column=3).value = f'{res[0] * 100.0:.2f}'
            work_sheet.cell(row=row_idx, column=4).value = f'{res[1] * 100.0:.2f}'
            work_sheet.cell(row=row_idx, column=5).value = f'{res[2] * 100.0:.2f}'
            work_sheet.cell(row=row_idx, column=6).value = f'{res[3]:.3f}'

        self.workbook.save(self.workbook_path)
        pass

    def _build_up_cmp_set(self, data_path, res_path, frm_len, mask_flag=False):
        cmp_set = []
        
        seq_num = len([x for x in res_path.glob('scene_*') if x.is_dir()])
        for seq_idx in range(seq_num):
            gt_scene_folder = data_path / f'scene_{seq_idx:04}'
            disp_gt_folder = gt_scene_folder / 'disp_gt'
            if not disp_gt_folder.exists():
                disp_gt_folder = gt_scene_folder / 'disp'
            res_scene_folder = res_path / f'scene_{seq_idx:04}'
            disp_res_folder = res_scene_folder / 'disp'

            for frm_idx in range(frm_len[seq_idx]):
                disp_gt_path = disp_gt_folder / f'disp_{frm_idx}.png'
                disp_res_path = disp_res_folder / f'disp_{frm_idx}.png'
                mask_path = gt_scene_folder / 'mask' / f'mask_{frm_idx}.png' if mask_flag else None

                if disp_gt_path.exists() and disp_res_path.exists():
                    cmp_set.append(tuple([disp_gt_path, disp_res_path, mask_path]))

        return cmp_set

    def _evaluate_exp_outs(self, cmp_set, prog_bar=None):
        res_array = []

        for disp_gt_path, disp_res_path, mask_path in cmp_set:

            disp_gt = plb.imload(disp_gt_path, scale=1e2)
            disp_res = plb.imload(disp_res_path, scale=1e2)
            mask = (disp_gt > 0.0).float()  # TODO
            if mask_path is not None:
                mask = plb.imload(mask_path)

            diff = (disp_gt - disp_res)
            diff_vec = diff[mask > 0.0]
            total_num = diff_vec.shape[0]
            err10_num = (torch.abs(diff_vec) > 1.0).float().sum() / total_num
            err20_num = (torch.abs(diff_vec) > 2.0).float().sum() / total_num
            err50_num = (torch.abs(diff_vec) > 5.0).float().sum() / total_num
            avg = torch.abs(diff_vec).sum() / total_num
            res_array.append(np.array([err10_num, err20_num, err50_num, avg]))

            if prog_bar is not None:
                prog_bar.update(1)

        res_array = np.stack(res_array, axis=0)
        res_avg = np.average(res_array, axis=0)
        return res_avg

    def sum_average(self, scene_name):
        src_work_sheet = self.workbook[scene_name]
        dst_work_sheet = self.workbook[f'{scene_name}-Sum']

        # Get all exp_tags & values
        start_row = 5
        raw_results = []
        for row_idx in range(start_row, src_work_sheet.max_row + 1):
            raw_results.append([
                src_work_sheet.cell(row_idx, 1).value,
                [
                    float(src_work_sheet.cell(row_idx, 3).value),
                    float(src_work_sheet.cell(row_idx, 4).value),
                    float(src_work_sheet.cell(row_idx, 5).value),
                    float(src_work_sheet.cell(row_idx, 6).value),
                ]
            ])

        # Group
        grouped_results = {}
        for exp_tag, eval_res in raw_results:
            exp_key = exp_tag
            if '_exp' in exp_tag:
                exp_key = exp_tag.split('_exp')[0]
            if exp_key not in grouped_results:
                grouped_results[exp_key] = []
            grouped_results[exp_key].append(np.array(eval_res))

        # Put into new sheet
        for i, exp_tag in enumerate(grouped_results):
            dst_work_sheet.cell(2 + i, 1).value = exp_tag
            exp_res = np.stack(grouped_results[exp_tag], axis=0)
            exp_avg = np.average(exp_res, axis=0)
            dst_work_sheet.cell(2 + i, 2).value = f'{exp_avg[0]:.2f}'
            dst_work_sheet.cell(2 + i, 3).value = f'{exp_avg[1]:.2f}'
            dst_work_sheet.cell(2 + i, 4).value = f'{exp_avg[2]:.2f}'
            dst_work_sheet.cell(2 + i, 5).value = f'{exp_avg[3]:.3f}'

        self.workbook.save(self.workbook_path)
        pass


# def draw_diff_viz(disp_gt, disp_map, mask):
#     diff = (disp_gt - disp_map)
#     step_err = torch.ones_like(diff)
#     step_err[torch.abs(diff) > 1.0] = 2.0
#     step_err[torch.abs(diff) > 2.0] = 3.0
#     step_err[torch.abs(diff) > 5.0] = 4.0
#     step_vis = plb.VisualFactory.err_visual(step_err, mask, max_val=4.0, color_map=cv2.COLORMAP_WINTER)
#     step_vis = cv2.cvtColor(plb.t2a(step_vis), cv2.COLOR_BGR2RGB)
#     return step_vis
#
#
# def draw_depth_viz(calib_file, json_file, depth_list, mask_map):
#     if len(depth_list) == 0:
#         return
#
#     config = ConfigParser()
#     config.read(str(calib_file), encoding='utf-8')
#     calib_para = config['Calibration']
#     wid, hei = plb.str2tuple(calib_para['img_size'], item_type=int)
#     wid, hei = 640, 480
#     fx, fy, dx, dy = plb.str2tuple(calib_para['img_intrin'], item_type=float)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=wid, height=hei)
#     vis.get_render_option().point_size = 1.0
#     # vis.get_render_option().load_from_json(str(json_file))
#     # param = None
#     if not json_file.exists():
#         depth_map = depth_list[0][0]
#         depth_viz = plb.DepthMapVisual(depth_map, (fx + fy) / 2.0, mask_map)
#         xyz_set = depth_viz.to_xyz_set()
#         xyz_set = xyz_set[mask_map.reshape(-1) == 1.0]
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz_set)
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30.0, max_nn=30)
#         )
#         pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_set) * 0.5)
#         vis.add_geometry(pcd)
#         vis.run()
#         param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#         o3d.io.write_pinhole_camera_parameters(str(json_file), param)
#         vis.clear_geometries()
#     ctr = vis.get_view_control()
#     param = o3d.io.read_pinhole_camera_parameters(str(json_file))
#
#     for depth_map, out_path in depth_list:
#         # Create mesh
#         depth_viz = plb.DepthMapVisual(depth_map, (fx + fy) / 2.0, mask_map)
#         xyz_set = depth_viz.to_xyz_set()
#         xyz_set = xyz_set[mask_map.reshape(-1) == 1.0]
#         np.savetxt(str(out_path.parent / f'{out_path.stem}.asc'), xyz_set,
#                    fmt='%.2f', delimiter=',', newline='\n', encoding='utf-8')
#
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz_set)
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30.0, max_nn=30)
#         )
#         pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_set) * 0.5)
#         vis.add_geometry(pcd)
#         ctr.convert_from_pinhole_camera_parameters(param)
#
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.01)
#
#         # vis.run()
#         # vis.get_render_option().save_to_json(str(json_file))
#
#         image = vis.capture_screen_float_buffer()
#         image = np.asarray(image).copy()
#         plb.imsave(out_path, image)
#
#         # Write result
#         vis.clear_geometries()
#         print(out_path.parent.name, out_path.name)
#
#     return


# def export_text_to_file(workbook, params):
#     # 顺序的几个
#     sheet_names = ['scene_00', 'scene_01', 'scene_02', 'scene_03']
#     pat_set = [7, 6, 5, 4, 3]
#     row_pair = [
#         ('7', 6),
#         ('6', 7),
#         ('5', 8),
#         ('4', 9),
#         ('3', 10),
#         ('23456', 11),
#         ('12457', 12),
#         ('1346', 16),
#         ('2457', 15),
#     ]
#     res = []
#     for pat_num, row_idx in row_pair:
#         for i, exp_name in enumerate(['Naive', 'NeRF', 'NeuS', 'Ours']):
#             title = f'{pat_num}-{exp_name}: & '
#             nums = []
#             col_s = i * 4 + 2
#             for x, sheet_name in enumerate(sheet_names):
#                 for bias in [2, 3]:
#                     val = workbook[sheet_name].cell(row=row_idx, column=col_s + bias).value
#                     if val is None:
#                         nums.append('~')
#                     else:
#                         if exp_name == 'Ours':
#                             val = '\\textbf{' + val + '}'
#                         nums.append(str(val))
#             num_str = ' & '.join(nums)
#             res.append(title + num_str + '\n')
#     with open(params['txt_name'], 'w+') as file:
#         file.writelines(res)

#     print(len(res))
#     pass


def copy_mad_to_path(src_path, dst_path):
    # Load csv file
    csv_name = list(src_path.glob('*.csv'))[0]
    loaded_info = []
    with open(str(csv_name), 'r', encoding='utf-8') as file:
        while True:
            res = file.readline()
            if res is None or res == '':
                break
            img_path = res.split(',')[0]
            scene_name = re.search('scene_\d+', img_path).group()
            frm_num = int(re.search('img_\d+', img_path).group().split('_')[1])
            loaded_info.append([scene_name, frm_num])

    # Get exp_tag
    exp_tags = [x.name for x in src_path.glob('*') if x.is_dir()]
    for exp_tag in exp_tags:
        for disparity_i, (scene_name, frm_num) in tqdm(enumerate(loaded_info), desc=exp_tag):
            # src_dispairty_file
            src_disp_file = src_path.joinpath(exp_tag,
                                              'disparities',
                                              f'disparity_{disparity_i}.png')
            # dst_disparity_file
            dst_disp_file = dst_path.joinpath(exp_tag,
                                              'output',
                                              csv_name.stem,
                                              'epoch_00001',
                                              scene_name,
                                              'disp',
                                              f'disp_{frm_num}.png')
            dst_disp_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(str(src_disp_file), str(dst_disp_file))


def main():
    # copy_mad_to_path(
    #     src_path=Path('/media/qiao/Videos/SLDataSet/OANet/52_RealData-out-mad'),
    #     dst_path=Path('/media/qiao/Videos/SLDataSet/OANet/52_RealData-out'),
    # )

    app = Evaluator(
        workbook='/media/qiao/Videos/SLDataSet/OANet/result.xlsx',
        flush_flag=True
    )
    # app.process_dataset('NonRigidReal', mask_flag=False)
    # app.sum_average('NonRigidReal')
    app.process_dataset('NonRigidVirtual', mask_flag=True)
    app.sum_average('NonRigidVirtual')


# def test():
#     mesh = o3d.io.read_triangle_mesh(r'C:\Users\qiao\Desktop\CVPR2023_Sub\scene_01\NeuS\pat5.ply')
#     mesh.compute_vertex_normals()
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=640, height=480)
#     vis.get_render_option().point_size = 1.0
#     vis.add_geometry(mesh)
#     vis.run()
#     image = vis.capture_screen_float_buffer()
#     image = np.asarray(image).copy()
#     plb.imsave('tmp.png', image)


# def main_sup():
#     main_folder = Path(r'C:\Users\qiao\Desktop\CVPR2023_Sub')
#     res_folder = main_folder / 'gif_set'
#     res_folder.mkdir(parents=True, exist_ok=True)
#     pcd_list = [
#         # (main_folder / f'scene_00' / 'gt.asc', res_folder / 'gt_00')
#     ]
#     for scene_idx in [0]:
#         for exp_set in ['NeuS']:  # [NeuS] 'Ours', 'GrayCode', 'Ours-sample', 'Ours-warp'
#             for exp_tag in ['pat5']:
#                 pcd_list.append(
#                     (main_folder / f'scene_{scene_idx:02}' / exp_set / f'{exp_tag}.ply',
#                      res_folder / f'{exp_set}_{scene_idx:02}_{exp_tag}')
#                 )
#     json_file = Path(r'C:\SLDataSet\20220907real\open3dvis.json')

#     wid, hei = 640, 480
#     total_num = 128
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=wid, height=hei)
#     vis.get_render_option().point_size = 1.0
#     ctr = vis.get_view_control()
#     param = o3d.io.read_pinhole_camera_parameters(str(json_file))

#     # Create pos set
#     pos_list = []
#     current_pos = param.extrinsic[:3, -1]
#     # xyz_set = np.loadtxt(pcd_list[0][0], delimiter=',', encoding='utf-8')
#     # mid_point = (xyz_set.max(axis=0) + xyz_set.min(axis=0)) * 0.5
#     # mid_point = - mid_point
#     # diff_vec = current_pos - mid_point
#     # diff_vec[1] = 0.0
#     # rad = np.linalg.norm(diff_vec)
#     rad = 100
#     for frm_idx in range(total_num):
#         angle = np.pi * 2 / (total_num * 0.5) * frm_idx
#         # pos = mid_point + np.array([
#         #     np.cos(angle) * rad,
#         #     0,
#         #     np.sin(angle) * rad
#         # ], dtype=mid_point.dtype)
#         pos = current_pos + np.array([
#             np.cos(angle) * rad,
#             np.sin(angle) * rad,
#             np.cos(angle * 1.5) * rad,
#         ])
#         cam_pos = plb.PosManager()
#         # cam_pos.set_from_look(pos, look, up)
#         cam_pos.set_rot(mat=param.extrinsic[:3, :3])
#         cam_pos.set_trans(pos)
#         pos_list.append(cam_pos)

#     for pcd_path, save_folder in pcd_list:
#         if not pcd_path.exists():
#             continue
#         # try:
#         #     xyz_set = np.loadtxt(str(pcd_path), delimiter=',', encoding='utf-8')
#         # except ValueError as e:
#         #     xyz_set = np.loadtxt(str(pcd_path), encoding='utf-8')
#         # xyz_set = xyz_set[xyz_set.sum(axis=1) > 1.0]
#         # pcd = o3d.geometry.PointCloud()
#         # pcd.points = o3d.utility.Vector3dVector(xyz_set)
#         # pcd.estimate_normals()
#         # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)) * 0.5)
#         pcd = o3d.io.read_triangle_mesh(str(pcd_path))
#         pcd.compute_vertex_normals()
#         pcd.compute_triangle_normals()
#         vis.add_geometry(pcd)
#         save_folder.mkdir(exist_ok=True, parents=True)
#         for frm_idx in range(total_num):
#             # Ctr
#             param.extrinsic = pos_list[frm_idx].get_4x4mat()
#             ctr.convert_from_pinhole_camera_parameters(param)
#             vis.poll_events()
#             vis.update_renderer()
#             time.sleep(0.01)
#             image = vis.capture_screen_float_buffer()
#             image = np.asarray(image).copy()
#             plb.imsave(save_folder / f'{frm_idx}.png', image)
#         vis.clear_geometries()

#     pass


# def draw_gifs():
#     main_folder = Path(r'C:\Users\qiao\Desktop\CVPR2023_Sub\Supplementary\data_scene0')
#     sub_folders = [x for x in main_folder.glob('*') if x.is_dir()]
#     for sub_folder in sub_folders:
#         gif_name = sub_folder.parent / f'{sub_folder.name}.gif'
#         writer = plb.GifWriter(gif_name, fps=2)
#         writer.run_folder(sub_folder)


# def clean_asc():
#     main_folder = Path(r'C:\Users\qiao\Desktop\CVPR2023_Sub\results')
#     asc_files = [x for x in main_folder.glob('*.asc')]
#     for asc_file in asc_files:
#         try:
#             xyz_set = np.loadtxt(str(asc_file), delimiter=',', encoding='utf-8')
#         except ValueError as e:
#             xyz_set = np.loadtxt(str(asc_file), encoding='utf-8')
#         xyz_set = xyz_set[xyz_set.sum(axis=1) > 1.0]
#         np.savetxt(str(asc_file.parent / f'{asc_file.stem}.xyz'), xyz_set,
#                    fmt='%.2f', delimiter=' ', newline='\n', encoding='utf-8')


if __name__ == '__main__':
    main()
    # main_sup()
    # draw_gifs()
