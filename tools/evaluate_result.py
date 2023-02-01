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
# import open3d as o3d

import pointerlib as plb


# - Coding Part - #

def evaluate(disp_gt, disp_map, mask_map, cell_set):
    
    def eval(gt, res, mask):
        diff = (gt - res)
        diff_vec = diff[mask > 0.0]
        total_num = diff_vec.shape[0]
        err10_num = (torch.abs(diff_vec) > 1.0).float().sum() / total_num
        err20_num = (torch.abs(diff_vec) > 2.0).float().sum() / total_num
        err50_num = (torch.abs(diff_vec) > 5.0).float().sum() / total_num
        avg = torch.abs(diff_vec).sum() / total_num
        return err10_num, err20_num, err50_num, avg
        
    if not isinstance(disp_gt, list):
        disp_gt = [disp_gt]
        disp_map = [disp_map]
        mask_map = [mask_map]
    
    frm_num = len(disp_gt)
    res_array = np.zeros([frm_num, 4])
    for i, (gt, res, mask) in enumerate(zip(disp_gt, disp_map, mask_map)):
        res_array[i] = eval(gt, res, mask)
    res_avg = np.average(res_array, axis=0)
    
    for i in range(3):
        cell_set[i].value = f'{res_avg[i] * 100.0:.2f}'
    cell_set[3].value = f'{res_avg[3]}'


def draw_diff_viz(disp_gt, disp_map, mask):
    diff = (disp_gt - disp_map)
    step_err = torch.ones_like(diff)
    step_err[torch.abs(diff) > 1.0] = 2.0
    step_err[torch.abs(diff) > 2.0] = 3.0
    step_err[torch.abs(diff) > 5.0] = 4.0
    step_vis = plb.VisualFactory.err_visual(step_err, mask, max_val=4.0, color_map=cv2.COLORMAP_WINTER)
    step_vis = cv2.cvtColor(plb.t2a(step_vis), cv2.COLOR_BGR2RGB)
    return step_vis


def process_scene(worksheet, params):
    data_path = Path(worksheet['B1'].value)
    res_path = Path(worksheet['B2'].value)

    # Config file
    config = ConfigParser()
    config.read(str(data_path / 'config.ini'), encoding='utf-8')
    seq_num = int(config['Data']['total_sequence'])
    frm_len = int(config['Data']['frm_len'])
    wid, hei = plb.str2tuple(config['Data']['img_size'], item_type=int)

    # Get experiments
    data_tags = [worksheet.cell(row=i + 1, column=1).value
                 for i in range(4, worksheet.max_row)]
    epoch_nums = [int(worksheet.cell(row=i + 1, column=3).value)
                  for i in range(4, worksheet.max_row)]

    # Load All GT
    gts_idx = []
    disp_gts = []
    mask_mats = []
    for seq_idx in range(seq_num):
        for frm_idx in range(frm_len):
            disp_gt_file = data_path / f'scene_{seq_idx:04}' / 'disp_gt' / f'disp_{frm_idx}.png'
            # if not disp_gt_file.exists():
            #    disp_gt_file = data_path / f'scene_{seq_idx:04}' / 'disp' / f'disp_{frm_idx}.png'
            if disp_gt_file.exists():
                disp_gt = plb.imload(disp_gt_file, scale=1e2)
                mask_mat = (disp_gt > 0.0).float()
                gts_idx.append((seq_idx, frm_idx))
                disp_gts.append(disp_gt)
                mask_mats.append(mask_mat)
    pass

    # Evaluate every experiments
    for data_i, (data_tag, epoch_num) in enumerate(zip(data_tags, epoch_nums)):
        print(data_tag)
        row_i, col_i = 5 + data_i, 4
        cell_set = [worksheet.cell(row=row_i, column=col_i + x) for x in range(4)]
        out_folder = res_path / data_tag / 'output' / data_path.name / f'epoch_{epoch_num:05}'
        worksheet.cell(row=row_i, column=2).value = str(out_folder)
        
        # Skip or not
        flag_all_filled = False
        for cell in cell_set:
            flag_all_filled = flag_all_filled and cell.value is not None
        if params['clear'] and not flag_all_filled:
            for cell in cell_set:
                cell.value = None
        if not params['recal'] and flag_all_filled:
            continue
        
        # Load disparity list
        disp_maps = []
        for seq_idx, frm_idx in gts_idx:
            disp_file = out_folder / f'scene_{seq_idx:04}' / 'disp' / f'disp_{frm_idx}.png'
            disp_maps.append(plb.imload(disp_file, scale=1e2))
            pass
        
        # Evaluate numbers
        evaluate(disp_gts, disp_maps, mask_mats, cell_set)
        
        # Draw diff map
        # diff_map = draw_diff_viz(depth_gt * depth_scale, depth_target * depth_scale, mask_gt)
        # plb.imsave(depth_target_path.parent / f'{exp}_diff.png', diff_map)
        
    pass


# def draw_depth_viz(calib_file, json_file, depth_list, mask_map):
#     if len(depth_list) == 0:
#         return

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

#     for depth_map, out_path in depth_list:
#         # Create mesh
#         depth_viz = plb.DepthMapVisual(depth_map, (fx + fy) / 2.0, mask_map)
#         xyz_set = depth_viz.to_xyz_set()
#         xyz_set = xyz_set[mask_map.reshape(-1) == 1.0]
#         np.savetxt(str(out_path.parent / f'{out_path.stem}.asc'), xyz_set,
#                    fmt='%.2f', delimiter=',', newline='\n', encoding='utf-8')

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz_set)
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30.0, max_nn=30)
#         )
#         pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_set) * 0.5)
#         vis.add_geometry(pcd)
#         ctr.convert_from_pinhole_camera_parameters(param)

#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.01)

#         # vis.run()
#         # vis.get_render_option().save_to_json(str(json_file))

#         image = vis.capture_screen_float_buffer()
#         image = np.asarray(image).copy()
#         plb.imsave(out_path, image)

#         # Write result
#         vis.clear_geometries()
#         print(out_path.parent.name, out_path.name)

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


def main():
    # Parameters
    params = {
        'recal': False,
        'clear': True,
        'workbook': '/media/qiao/Videos/SLDataSet/OANet/result.xlsx',
        # 'txt_name': '/media/qiao/Videos/SLDataSet/OANet/ForLatex.txt'
    }
    target_sheet_names = None

    # Load workbook and process sheets
    workbook = openpyxl.load_workbook(str(params['workbook']))

    if target_sheet_names is None:
        target_sheet_names = workbook.get_sheet_names()
    for scene_name in target_sheet_names:
        # print(scene_name)
        process_scene(workbook[scene_name], params)

    # export_text_to_file(workbook, params)

    workbook.save(str(params['workbook']))


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
    main_folder = Path(r'C:\Users\qiao\Desktop\CVPR2023_Sub\results')
    asc_files = [x for x in main_folder.glob('*.asc')]
    for asc_file in asc_files:
        try:
            xyz_set = np.loadtxt(str(asc_file), delimiter=',', encoding='utf-8')
        except ValueError as e:
            xyz_set = np.loadtxt(str(asc_file), encoding='utf-8')
        xyz_set = xyz_set[xyz_set.sum(axis=1) > 1.0]
        np.savetxt(str(asc_file.parent / f'{asc_file.stem}.xyz'), xyz_set,
                   fmt='%.2f', delimiter=' ', newline='\n', encoding='utf-8')



if __name__ == '__main__':
    # test()
    main()
    # main_sup()
    # draw_gifs()
