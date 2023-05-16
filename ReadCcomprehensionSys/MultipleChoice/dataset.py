# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/17 8:27 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : dataset.py 
# @Software: PyCharm

from torch.utils.data import Dataset

# 定义ChIDDataset类，继承自torch.utils.data.Dataset
class ChIDDataset(Dataset):
    # 定义初始化函数，输入参数为数据data
    def __init__(self, data):
        # 将数据存储到实例变量self.data中
        self.data = data

    # 定义len方法，返回数据的长度
    def __len__(self):
        return len(self.data)

    # 定义getitem方法，根据给定的索引idx返回相应的数据样本
    def __getitem__(self, idx):
        # 获取对应索引idx的数据样本
        example = self.data[idx]
        # 返回数据样本的input_ids、attention_mask、answer和candidate
        return example["input_ids"], example["attention_mask"], example["answer"], example["candidate"]
