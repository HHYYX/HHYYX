# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/17 8:29 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : trainer.py 
# @Software: PyCharm

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

torch.cuda.empty_cache()
import torch
from MultipleChoice.dataset import ChIDDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


# 定义ChIDTrainer类
class ChIDTrainer:
    # 定义初始化函数，输入参数包括model、train_data、dev_data、device、epochs、batch_size、learning_rate、warmup_steps和output_dir
    def __init__(self, model, train_data, dev_data, device, epochs, batch_size, learning_rate, warmup_steps,
                 output_dir):
        # 将各个参数存储到实例变量中
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir

        # 定义train方法，用于模型训练

    def train(self):
        # 创建训练集和开发集的dataloader
        train_dataset = ChIDDataset(self.train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)

        dev_dataset = ChIDDataset(self.dev_data)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=self.batch_size)

        # 定义优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=len(train_dataloader) * self.epochs)

        # 将模型移动到指定的设备上
        self.model.to(self.device)

        # 在训练循环开始前添加此行，用于存储最低验证损失
        lowest_eval_loss = float("inf")

        # 开始训练循环
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            # 遍历训练集的dataloader，并对每个batch进行训练
            for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}"):
                # 从batch中获取输入数据、标签和备选成语
                input_ids, attention_mask, labels, candidates = batch
                # 将输入数据reshape成(batch_size, num_choices, sequence_length)的形式
                input_ids = input_ids.view(self.batch_size, -1, input_ids.size(1))
                attention_mask = attention_mask.view(self.batch_size, -1, attention_mask.size(1))
                labels = labels.view(self.batch_size, -1)
                # 对标签执行argmax操作
                labels = torch.argmax(labels, dim=-1)
                # 将输入数据、标签等数据移动到指定的设备上
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(
                    self.device), labels.to(self.device)

                # 将模型的梯度清零，然后执行前向传播、计算损失、反向传播和梯度更新操作
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()

            # 计算平均训练损失并输出
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Average train loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

            # 执行模型的验证操作，并计算验证损失和验证准确率
            eval_loss, eval_accuracy = self.evaluate(dev_dataloader)
            print(f"Validation loss for epoch {epoch + 1}: {eval_loss:.4f}")
            print(f"Validation accuracy for epoch {epoch + 1}: {eval_accuracy:.4f}")

            # 在每个epoch的训练和评估后添加以下代码
            if eval_loss < lowest_eval_loss:
                lowest_eval_loss = eval_loss
                print(f"Saving best model with validation loss: {lowest_eval_loss:.4f}")
                self.model.save_pretrained(self.output_dir)

    # 定义evaluate方法，用于模型验证
    def evaluate(self, dataloader):
        # 将模型切换到评估模式
        self.model.eval()

        # 定义变量total_loss、total_correct和total_samples，用于统计损失、正确样本数和总样本数
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 遍历验证集的dataloader，并对每个batch进行验证
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 从batch中获取输入数据、标签和备选成语
            input_ids, attention_mask, labels, candidates = batch
            # 将输入数据reshape成(batch_size, num_choices, sequence_length)的形式
            input_ids = input_ids.view(self.batch_size, -1, input_ids.size(1))
            attention_mask = attention_mask.view(self.batch_size, -1, attention_mask.size(1))
            labels = labels.view(self.batch_size, -1)
            # 对标签执行argmax操作
            labels = torch.argmax(labels, dim=-1)
            # 将输入数据、标签等数据移动到指定的设备上
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(
                self.device), labels.to(self.device)

            # 在验证时不更新梯度，只进行前向传播、计算损失和计算准确率
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

        # 计算平均验证损失和验证准确率并返回
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
