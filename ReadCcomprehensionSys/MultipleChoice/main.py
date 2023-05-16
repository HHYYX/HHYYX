# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 16:01 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : main.py 
# @Software: PyCharm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.model_selection import train_test_split
# 清除 GPU 缓存
import json
from transformers import BertTokenizerFast
import torch
torch.cuda.empty_cache()
import torch
from MultipleChoice.data_format import load_data, load_and_process_data, preprocess_data
from MultipleChoice.trainer import ChIDTrainer
from transformers import BertForMultipleChoice


if __name__ == "__main__":

    # 设定相关参数
    train_content_path = "../data/ChID/train.json"
    train_answer_path = "../data/ChID/train_answer.json"
    # dev_content_path = "../data/ChID/dev.json"
    # dev_answer_path = "../data/ChID/dev_answer.json"

    epochs = 10
    batch_size = 8
    learning_rate = 2e-5
    warmup_steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = BertForMultipleChoice.from_pretrained("../model/chinese-bert-wwm")
    tokenizer = BertTokenizerFast.from_pretrained("../model/chinese-bert-wwm")
    # 加载和预处理数据
    # train_data, dev_data = load_and_process_data(train_content_path, train_answer_path, dev_content_path,
    #                                              dev_answer_path, tokenizer)
    # 加载和预处理数据
    train_content_data, train_answer_data = load_data(train_content_path, train_answer_path)
    train_data = preprocess_data(train_content_data, train_answer_data, tokenizer)
    # 将训练数据集划分为训练集和验证集
    train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=42)
    # 设定模型保存的目录
    output_dir = "../output/multi_choice/best_model"

    # 实例化 ChIDTrainer
    trainer = ChIDTrainer(model, train_data, dev_data, device, epochs, batch_size, learning_rate, warmup_steps, output_dir)

    # 开始训练
    trainer.train()


