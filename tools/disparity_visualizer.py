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


# - Coding Part - #
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


def main():
    # copy_visualization_res(
    #     data_path=Path('/media/qiao/Videos/SLDataSet/OANet/52_RealData'),
    #     res_path=Path('/media/qiao/Videos/SLDataSet/OANet/52_RealData-out'),
    #     img_set=[
    #         ('scene_0000', 255),
    #         ('scene_0001', 255),
    #         ('scene_0002', 255),
    #         ('scene_0003', 255)
    #     ],
    #     exp_set=[
    #         'mad-off_exp3',
    #         'mad-lcn_exp1',
    #         # 'asn-eval',
    #         # 'ctd-eval',
    #         # 'mad_exp0',
    #         # 'tide-eval',
    #         # 'oade-pfwom_exp1',
    #         # 'oade-phpfwom_exp1',
    #         # 'oade-ph_exp1',
    #         # 'oade-pf_exp1',
    #         # 'oade-frm0_exp1',
    #         # 'oade-frm32_exp1',
    #         # 'oade-frm64_exp1',
    #         # 'oade-frm96_exp1',
    #         # 'oade-frm128_exp1',
    #         # 'oade-frm160_exp1',
    #         # 'oade-frm192_exp1',
    #         # 'oade-frm224_exp1',
    #     ],
    #     out_path=Path('/media/qiao/Videos/SLDataSet/OANet/52_RealData-vis')
    # )
    copy_visualization_res(
        data_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualData'),
        res_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualData-out'),
        img_set=[
            # ('scene_0000', 143),
            # ('scene_0001', 87),
            # ('scene_0000', 95),
            # ('scene_0003', 239),
            # ('scene_0002', 159),
            ('scene_0003', 495),
        ],
        exp_set=[
            'mad-off_exp1',
            'mad-ssmi_exp1',
            # 'asn-eval',
            # 'ctd-eval',
            # 'mad_exp1',
            # 'tide-eval',
            # 'oade-phpfwom_exp1',
        ],
        out_path=Path('/media/qiao/Videos/SLDataSet/OANet/31_VirtualData-vis')
    )
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