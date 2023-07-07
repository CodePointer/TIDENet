# TIDENet

The PyTorch implementation of TIDE.

If you find our code useful, please cite:

```
@article{qiao2022tide,
    title={{TIDE}: Temporally Incremental Disparity Estimation via Pattern Flow in Structured Light System},
    author={Rukun Qiao and Hiroshi Kawasaki and Hongbin Zha},
    journal=RAL,
    year={2022},
    volume={7},
    pages={5111-5118},
}
```

## Installation

The project is implemented based on PyTorch and OpenCV. 

You can install the required packages by:

```
pip install -r requirments.txt
```

## Datasets:

Please download the dataset from here:

- [Real-world Rigid Motion](https://drive.google.com/file/d/1gQfYVir8dSnWj_CB7pGNOzIrGZURVM27/view?usp=sharing), used in TIDE with groundtruth.
- [Real-world Non-rigid Motion](), used in TIDE without groundtruth. Only raw disparity from RealSense sensor is given.

After downloaded the dataset, you can add the directory to the parameters under `params` folder. With the given dataset, you should be able to run the code without any preprocessing. 

### Use Your Own Dataset

You can also use your own dataset. Please follow the follow the similar dataset directory scheme or rewrite the `models/img_clip_dataset.py`. Please refer to the followings for further preprocessing.


### Calibration parameters

All the calibrated parameters are stored as:

1. Calibrated system -> config.ini
2. Projected pattern in rectified system -> rect_pattern.png
3. Detect pattern center & offline graph -> pat_info.pt

You have to provide the first two file if you use your own calibrated parameters. For the third one, you can use a provided script `tools/pattern_compte.py`. Change the given pattern path to compute the `pat_info.pt`.

```
export PYTHONPATH=.  # <set PYTHONPATH=.> if under windows os
python tools/draw_mask_center.py
```

After finished, please copy the generated `pat_info.pt` to the dataset folder.

### Detect dot center

The dot center detection process is implemented by OpenCV, blob detection. This preprocessing is provided in script `tools/pattern_compute.py`.

```
export PYTHONPATH=.  # <set PYTHONPATH=.> if under windows os
python tools/pattern_compute.py
```

## Pretrain

You can download the pretrained model from [Here](https://drive.google.com/file/d/1y-jeuQ_HGmRdcjhEylO-QPuw9v605hPP/view?usp=sharing).

You can also train the network by your self. First, pretrain the initial module:

```
python main.py --config ./params/init_train.ini
```

Then with the given init module, pretrain the TIDE network:

```
python main.py --config ./params/tide_train.ini
```

## Evaluation and Online Adaptation

Evaluate only, without any adaptation:

```
python main.py --config ./params/tide_eval.ini
```
