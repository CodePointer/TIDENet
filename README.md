# TIDENet

The PyTorch implementation of TIDE & Online Adaptation Version.

If you find our code useful, please cite:

```
@article{qiao2022tide,
    title={{TIDE}: Temporally Incremental Disparity Estimation via Pattern Flow in Structured Light System},
    author={Rukun Qiao and Hiroshi Kawasaki and Hongbin Zha},
    journal={RAL},
    year={2022},
    volume={7},
    pages={5111-5118},
}

@inproceedings{qiao2023online,
  title={Online Adaptive Disparity Estimation for Dynamic Scenes in Structured Light Systems},
  author={Rukun Qiao and Hiroshi Kawasaki and Hongbin Zha},
  booktitle={IROS}, 
  year={2023}
}
```

## Step 0: Installation

The project is implemented based on PyTorch and OpenCV. 

You can install the required packages by:

```
pip install -r requirments.txt
```

Our code is using `python==3.6` and `torch==1.8.1`. Higher version of python may be fine, but please make sure your version is later than 3.6, for we have a lot of f-strings in our codes.

P.S: `Open3D` and `openpyxl` is only used for visualization.


## Step 1: Prepare you datasets

### Download the datasets:

Please download the dataset from here:

- [Rigid Synthetic](https://drive.google.com/file/d/1aqIqRtgtgNH5sj8fWjXDgHMWngEf6vO_/view?usp=drive_link). Rigid synthetic scene with moving camera.
- [Rigid Real](https://drive.google.com/file/d/1oAbUO-E_aQrIngG9Bg1QdOdQRersMJlZ/view?usp=drive_link). Groundtruth for object only.
- [Non-rigid Synthetic](https://drive.google.com/file/d/1udqZwfrH0RhFa14lf4h1KeeXbO68J6pi/view?usp=drive_link), A dataset rendered based on SceneFlow.
- [Non-rigid Real](https://drive.google.com/file/d/1XF2KcpY4kVzAMwsJ2EtZlg8EWoX857fa/view?usp=drive_link), with sparse groundtruth for last frame.

Please put the dataset under `./data` folder:

```
- {ProjectFolder}
    ...
    |- data
        |- Non-rigid-Real
            |- pat
            |- scene_0000
            ...
            |- config.ini
        ...
    ...
```

### Or Use Your Own Dataset

You can also use your own dataset for evaluation and training. Please follow the follow the similar dataset directory scheme or rewrite the `models/img_clip_dataset.py`. You may also need some preprocessing for your dataset.

#### Calibration parameters

All the calibrated parameters are stored as:

1. Calibrated system -> `config.ini`
2. Projected pattern in rectified system -> `./pat/pat_0.png`
3. Detect pattern center & offline graph -> `pat_info.pt`

You have to provide the first two file if you use your own calibrated parameters. For the third one, you can use a provided script `tools/pattern_compte.py`. Change the given pattern path to compute the `pat_info.pt`. Please set the data path under this script and run:

```
export PYTHONPATH=.  # <set PYTHONPATH=.> if under windows os
python tools/pattern_compte.py
```

After finished, please copy the generated `pat_info.pt` to the dataset folder.

Notice that our framework requires same calibration paramters and patterns for training and evaluating.

#### Detect dot center

The dot center detection process is implemented by OpenCV, blob detection. This preprocessing is provided in script `tools/draw_mask_center.py`. Please set the data path under this script and run:

```
export PYTHONPATH=.  # <set PYTHONPATH=.> if under windows os
python tools/draw_mask_center.py
```


## Step 2: Pretrained Model

You can download the pretrained model from [Here](https://drive.google.com/file/d/1GiGtW3pDEZ1TzQpcbvjCAFQfsu8f21IE/view?usp=drive_link). Please put the models under `./data` folder like this:

```
- {ProjectFolder}
    ...
    |- data
        |- Pretrained-Models
            |- TIDE_Ft_e0.pt
            |- TIDE_NtH_e0.pt
            |- TIDE_Up_e0.pt
        ...
    ...
```

You can also train the network by your self. Just prepare the pretrained dataset and set the `train_dir` in `./params/tide_train.ini`. Then run the script:

```
python main.py --config ./params/tide_train.ini
```

## Step 3: Evaluation and Online Adaptation

Please set the `train_dir` for online learning and set the `test_dir` for evaluation. Also make sure the `model_dir` is correct:

```
...
test_dir = ./data/Non-rigid-Real
model_dir = ./data/Pretrained-Models
...
```

Evaluate only, without any adaptation:

```
chmod +x ./scripts/run_eval.sh
./run_eval.sh
```

Evaluation with online adapataion
```
chmod +x ./scripts/run_online.sh
./run_online.sh
```

Results will be saved under `out_dir`.

## Step 4: Analyze the Results

We provide two files for quantitative evaluation (`./tools/evaluate_result.py`) and qualitative visualization (`./tools/dispairty_visualizer.py`).

For quantitative evaluation, please prepare the excel summary file and set the path in `evaluate_result.py`. There are several requirements for the excel file:

- The data path and output path must be set in ceil `B2` and `B3`.
- The summary results will start from line 5.
- Different sheet name indicates different datasets, but the name is not necessarily the same.
- We also support evaluating multiple times and get the average of them. Just set your experiements name as `{some_tag_here}_exp1, {some_tag_here}_exp2, ...`. Please refer to the script for more details.

We have an example excel file under the project folder. Please check that if you need it.

For qualitative visualization, please change the path in `evaluate_result.py`. Notice the visualization of point cloud is implemented based on Open3D and require an monitor to finished the rendering. If you don't have a monitor, you may need to complie the Headless Rendering for Open3D module.
