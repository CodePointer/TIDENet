# -*- coding: utf-8 -*-

# @Time:      2023/02/15 11:14
# @File:      dispairty_visualizer.py
# @Software:  VSCode
# @Description:
#   This file is used for disparity map evaluation.
#   Notice the coordinate for open3d is: left x, up y, left handed.

# - Package Imports - #
from pathlib import Path
import shutil
import numpy as np
from configparser import ConfigParser
import open3d as o3d
import time
import torch
import cv2
from tqdm import tqdm

import pointerlib as plb


# - Coding Part - #
def _disp2pcd(disp_file, pcd_file, calib_para, mask_file=None):
    # disp -> depth
    fb = float(calib_para['focal_len']) * float(calib_para['baseline'])
    disp = plb.imload(disp_file, scale=1e2, flag_tensor=False)
    depth_map = fb / disp
    depth_map[disp == 0.0] = 0.0

    # depth -> xyz_set
    fx, fy, dx, dy = plb.str2tuple(calib_para['img_intrin'], item_type=float)
    if mask_file is None:
        mask = (depth_map > 0).astype(np.float32)
    else:
        mask = plb.imload(mask_file, flag_tensor=False)
    depth_viz = plb.DepthMapConverter(depth_map, (fx + fy) / 2.0)
    xyz_set = depth_viz.to_xyz_set()
    xyz_set = xyz_set[mask.reshape(-1) > 0.0]
    xyz_set = xyz_set[xyz_set[:, -1] > 10.0, :]
    xyz_set = xyz_set[xyz_set[:, -1] < 1200.0, :]

    # Save to pcd file
    np.savetxt(str(pcd_file), xyz_set, fmt='%.2f', delimiter=',',
               newline='\n', encoding='utf-8')

    pass


def _pcd2vis(pcd_file, vis_file, calib_para, visible=False):
    wid, hei = plb.str2tuple(calib_para['img_size'], item_type=int)
    fx, fy, dx, dy = plb.str2tuple(calib_para['img_intrin'], item_type=float)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=wid, height=hei, visible=visible)
    inf_pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=wid, height=hei,
        fx=fx, fy=fy, cx=dx, cy=dy
    )
    extrinsic_mat = np.array([
        [1.0, 0.0, 0.0, -30.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 50.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = inf_pinhole
    camera.extrinsic = extrinsic_mat
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)

    # Create pcd
    xyz_set = np.loadtxt(str(pcd_file), delimiter=',', encoding='utf-8')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_set)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30.0, max_nn=30)
    )
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_set) * 0.5)

    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)

    # vis.run()

    img = vis.capture_screen_float_buffer()
    img = np.asarray(img).copy()

    vis.clear_geometries()
    vis.destroy_window()

    # Save to vis file
    plb.imsave(vis_file, img)


def _draw_step_vis(gt_file, disp_file, step_file, mask_file=None):
    disp_gt = plb.imload(gt_file, scale=1e2)
    mask = (disp_gt > 0).float()
    if mask_file is not None:
        mask = plb.imload(mask_file)
    disp_res = plb.imload(disp_file, scale=1e2)

    diff = (disp_gt - disp_res)

    step_err = torch.ones_like(diff)
    step_err[torch.abs(diff) > 1.0] = 2.0
    step_err[torch.abs(diff) > 2.0] = 3.0
    step_err[torch.abs(diff) > 5.0] = 4.0
    step_vis = plb.VisualFactory.err_visual(step_err, mask, max_val=4.0, color_map=cv2.COLORMAP_WINTER)
    step_vis = cv2.cvtColor(plb.t2a(step_vis), cv2.COLOR_BGR2RGB)

    # Save to step res
    plb.imsave(step_file, step_vis)


def _refill_step_vis(gt_file, step_file):
    """For dot-based visualization. The original step is too small for dot-based gt."""
    disp_gt = plb.imload(gt_file, scale=1e2, flag_tensor=False)
    h_set, w_set = np.where(disp_gt > 0)
    step_img = plb.imload(step_file, flag_tensor=False)
    step_color = (step_img[h_set, w_set, :] * 255).astype(np.uint8)
    new_mat = np.zeros_like(step_img).astype(np.uint8)
    for i in range(step_color.shape[0]):
        center = tuple([w_set[i], h_set[i]])
        color = tuple(step_color[i].tolist())
        new_mat = cv2.circle(new_mat, center, 3, color, thickness=-1)
    plb.imsave(step_file, new_mat, scale=1.0)


def copy_visualization_res(data_path, res_path, img_set, exp_set, out_path):
    
    for scene_folder_name, frm_idx in img_set:
        out_folder = out_path / f'{scene_folder_name}-{frm_idx}'
        out_folder.mkdir(exist_ok=True, parents=True)

        # Copy ground-truth & image & mask
        shutil.copy(
            str(data_path / scene_folder_name / 'disp' / f'disp_{frm_idx}.png'),
            str(out_folder / 'disp_gt.png')
        )
        shutil.copy(
            str(data_path / scene_folder_name / 'img' / f'img_{frm_idx}.png'),
            str(out_folder / 'img.png')
        )
        shutil.copy(
            str(data_path / scene_folder_name / 'mask' / f'mask_{frm_idx}.png'),
            str(out_folder / 'mask.png')
        )

        # Copy res from all the exp_set
        for exp_tag in exp_set:
            sub_folders = [x for x in (res_path / exp_tag / 'output' / data_path.name).glob('*') if x.is_dir()]
            exp_res_path = sorted(sub_folders)[-1]
            shutil.copy(
                str(exp_res_path / scene_folder_name / 'disp' / f'disp_{frm_idx}.png'),
                str(out_folder / f'disp_{exp_tag}.png')
            )

        print(f'{out_folder.name} finished.')
    
    pass


def visualize_result(data_folder, output_folder, mask=True, refill=False):
    # Load config
    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    calib_para = config['RectCalib']

    # Folder_set
    scene_names = sorted([x.name for x in output_folder.glob('*') if x.is_dir()])
    for scene_name in scene_names:
        gt_folder = data_folder / scene_name / 'disp'
        disp_folder = output_folder / scene_name / 'disp'
        mask_folder = data_folder / scene_name / 'mask'
        frm_len = len(list(disp_folder.glob('disp_*.png')))

        # For output
        vis_folder = output_folder / scene_name / 'vis'
        vis_folder.mkdir(exist_ok=True)

        for frm_idx in tqdm(range(frm_len), desc=scene_name):

            # disp -> pcd
            disp_file = disp_folder / f'disp_{frm_idx}.png'
            mask_file = mask_folder / f'mask_{frm_idx}.png' if mask else None
            pcd_file = vis_folder / f'pcd_{frm_idx}.xyz'

            if not pcd_file.exists():
                _disp2pcd(disp_file, pcd_file, calib_para, mask_file)
            
            # pcd -> vis
            vis_file = vis_folder / f'vis_{frm_idx}.png'
            if not vis_file.exists():
                _pcd2vis(pcd_file, vis_file, calib_para, True)
            
            # disp, gt -> step
            gt_file = gt_folder / f'disp_{frm_idx}.png'
            step_file = vis_folder / f'step_{frm_idx}.png'
            if gt_file.exists() and not step_file.exists():
                _draw_step_vis(gt_file, disp_file, step_file, mask_file)
                if refill:
                    _refill_step_vis(gt_file, step_file)

    pass


def main():
    
    data_set = '3_Non-rigid-Real'
    exp_name = 'oade-online-wonbr2_exp2'
    epoch_num = 1
    mask_flag = False
    refill = True

    data_folder = Path(f'./data/{data_set}')
    output_folder = Path(f'./output/{data_set}/{exp_name}/output/{data_set}/epoch_{epoch_num:05}')
    visualize_result(data_folder, output_folder, mask_flag, refill)

    # copy_visualization_res(
    #     data_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualDataEval'),
    #     res_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualDataEval-out'),
    #     img_set=[
    #         ('scene_00', 255),
    #         # ('scene_0001', 87),
    #         ('scene_04', 255)
    #         # ('scene_0002', 159),
    #         # ('scene_0003', 495)
    #     ],
    #     exp_set=[
    #         'asn-eval',
    #         'ctd-eval',
    #         'mad-lcn_exp1',
    #         'mad-off',
    #         'tide-eval',
    #         'oade-phpfwom_exp1',
    #     ],
    #     out_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualDataEval-vis')
    # )

    pass


if __name__ == '__main__':
    main()
