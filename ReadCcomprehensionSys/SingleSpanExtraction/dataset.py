# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 10:34 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : dataset.py 
# @Software: PyCharm

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset


class CMRCDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data  # 数据列表
        self.tokenizer = tokenizer  # 分词器
        self.max_length = max_length  # 最大长度

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, index):
        item = self.data[index]  # 获取指定索引处的数据
        context = item['context']  # 获取上下文文本
        question = item['question']  # 获取问题文本
        answer_start_char = item['answer_start']  # 获取答案开始字符位置
        answer_text = item['answer']  # 获取答案文本

        # 编码上下文和问题，并添加特殊标记，以获取输入、注意力掩码和偏移映射等张量
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation="only_second",  # 只对上下文进行截断
            padding="max_length",  # 对输入进行填充，使得它们的长度一致
            return_attention_mask=True,
            return_tensors='pt',  # 返回 PyTorch 张量
            return_offsets_mapping=True  # 添加这一行以返回偏移映射
        )

        # 获取 token 级别的答案开始和结束位置
        answer_end_char = answer_start_char + len(answer_text)  # 获取答案结束字符位置
        offsets = inputs['offset_mapping'].squeeze()  # 获取偏移映射，并进行压缩
        answer_start_token_candidates = (offsets[:, 0] == answer_start_char).nonzero(as_tuple=True)[0]  # 获取可能的起始位置
        if answer_start_token_candidates.size(0) > 0:
            answer_start_token = answer_start_token_candidates[0].item()  # 获取真正的起始位置

            answer_end_token_candidates = (offsets[:, 1] == answer_end_char).nonzero(as_tuple=True)[0]  # 获取可能的结束位置
            if answer_end_token_candidates.size(0) > 0:
                answer_end_token = answer_end_token_candidates[0].item()  # 获取真正的结束位置
            else:
                answer_end_token = (offsets[:,
                                    1] - answer_end_char).abs().argmin().item()  # 如果没有可能的结束位置，则选择最接近答案结束字符位置的 token 作为结束位置

        else:
            answer_start_token = -1  # 如果没有可能的起始位置，则将起始位置设为 -1
            answer_end_token = -1  # 如果没有可能的结束位置，则将结束位置设为 -1

        input_ids = inputs['input_ids'].squeeze()  # 压缩输入张量
        attention_mask = inputs['attention_mask'].squeeze()  # 压缩注意力掩码张量

        start_positions = torch.tensor(answer_start_token, dtype=torch.long)  # 将起始位置转换为张量
        end_positions = torch.tensor(answer_end_token, dtype=torch.long)  # 将结束位置转换为张量

        return input_ids, attention_mask, start_positions, end_positions, offsets
