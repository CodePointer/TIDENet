# -*- coding: utf-8 -*-

# @Time:      2022/9/20 12:56
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      base_dataset.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from torch.utils.data import Dataset


# - Coding Part - #
class BaseDataset(Dataset):
    """Base dataset for training. Including a dataset tag for data saving."""
    def __init__(self, dataset_tag):
        super(BaseDataset, self).__init__()
        self.dataset_tag = dataset_tag

    def __getitem__(self, item):
        raise NotImplementedError

    def get_tag(self):
        return self.dataset_tag
