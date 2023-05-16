# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 10:09 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : main.py
# @Software: PyCharm


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from SingleSpanExtraction.dataset import CMRCDataset
from SingleSpanExtraction.data_format import load_data
from SingleSpanExtraction.trainer import CMRCTrainer

if __name__ == "__main__":
    train_data = load_data('../data/CMRC/cmrc2018_train.json')  # 加载训练集数据
    val_data = load_data('../data/CMRC/cmrc2018_trial.json')  # 加载验证集数据

    # 设置参数
    model_name = '../model/chinese-bert-wwm'  # 预训练 BERT 模型的名称
    tokenizer = BertTokenizerFast.from_pretrained(model_name)  # 创建 tokenizer 实例
    model = BertForQuestionAnswering.from_pretrained(model_name)  # 创建 BertForQuestionAnswering 模型实例

    # 将训练集和验证集数据转换为 CMRC2018Dataset 实例
    train_dataset = CMRCDataset(train_data, tokenizer)
    val_dataset = CMRCDataset(val_data, tokenizer)

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)

    # 检查设备是否可用，并将模型移动到指定的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建 CMRCTrainer 实例
    trainer = CMRCTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_examples=val_data,
        device=device,
        output_dir='../output/',
        epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )

    # 开始训练模型
    trainer.train()
