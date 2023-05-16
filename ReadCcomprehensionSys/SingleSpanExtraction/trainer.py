# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 10:34 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : trainer.py 
# @Software: PyCharm

import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer


class CMRCTrainer():
    def __init__(self, model, train_dataset, val_dataset, val_examples, device, output_dir, epochs=3, batch_size=8,
                 learning_rate=2e-5):
        self.model = model  # 模型
        self.train_dataset = train_dataset  # 训练集
        self.val_dataset = val_dataset  # 验证集
        self.device = device  # 设备
        self.output_dir = output_dir  # 输出目录
        self.epochs = epochs  # 训练轮数
        self.batch_size = batch_size  # 批次大小
        self.learning_rate = learning_rate  # 学习率
        self.val_examples = val_examples  # 验证例子
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集数据加载器
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集数据加载器
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)  # 优化器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * epochs
        )  # 学习率调度器，以实现学习率的线性下降

    def train(self):
        self.model.to(self.device)  # 将模型移动到指定的设备上

        best_similarity = 0.0  # 用于保存最佳相似度得分的变量

        for epoch in range(self.epochs):
            self.model.train()  # 设置模型为训练模式
            train_loss = 0.0  # 用于保存训练损失的变量
            for batch in tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}"):  # 遍历训练集的数据加载器
                input_ids, attention_mask, start_positions, end_positions, _ = batch  # 获取输入、注意力掩码和答案位置等张量
                input_ids, attention_mask, start_positions, end_positions = input_ids.to(
                    self.device), attention_mask.to(self.device), start_positions.to(self.device), end_positions.to(
                    self.device)  # 将张量移动到指定的设备上

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions, end_positions=end_positions)  # 获取模型的输出
                loss = outputs.loss  # 获取损失
                loss.backward()  # 反向传播

                train_loss += loss.item()  # 累加训练损失

                self.optimizer.step()  # 更新模型参数
                self.scheduler.step()  # 更新学习率
                self.optimizer.zero_grad()  # 清空梯度

            avg_train_loss = train_loss / len(self.train_loader)  # 计算平均训练损失
            print(f"Average train loss for epoch {epoch + 1}: {avg_train_loss:.4f}")  # 打印平均训练损失

            val_similarity = self.evaluate()  # 在验证集上评估模型
            print(f"Validation similarity scores for epoch {epoch + 1}: {val_similarity:.4f}")  # 打印验证集的相似度得分

            if val_similarity > best_similarity:  # 如果当前模型的相似度得分高于最佳相似度得分
                best_similarity = val_similarity  # 更新最佳相似度得分
                self.save_model()  # 保存当前模型

    def evaluate(self):
        self.model.eval()  # 设置模型为评估模式
        true_answers = []  # 用于保存真实答案的列表
        pred_answers = []  # 用于保存预测答案的列表

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):  # 遍历验证集的数据加载器
                input_ids, attention_mask, start_positions, end_positions, _ = batch  # 获取输入、注意力掩码和答案位置等张量
                input_ids = input_ids.to(self.device)  # 将输入张量移动到指定的设备上
                attention_mask = attention_mask.to(self.device)  # 将注意力掩码张量移动到指定的设备上
                start_positions = start_positions.to(self.device)  # 将答案开始位置张量移动到指定的设备上
                end_positions = end_positions.to(self.device)  # 将答案结束位置张量移动到指定的设备上

                outputs = self.model(input_ids, attention_mask=attention_mask)  # 获取模型的输出
                start_logits, end_logits = outputs.start_logits, outputs.end_logits  # 获取起始位置和结束位置的预测概率
                start_preds = torch.argmax(start_logits, dim=-1).cpu().numpy()  # 获取起始位置的预测
                end_preds = torch.argmax(end_logits, dim=-1).cpu().numpy()  # 获取结束位置的预测

                for i in range(len(start_preds)):  # 遍历当前批次的每个样本
                    pred_start = start_preds[i]  # 获取起始位置的预测
                    pred_end = end_preds[i]  # 获取结束位置的预测
                    pred_answer = self.val_examples[idx * self.batch_size + i]['context'][
                                  pred_start:pred_end + 1]  # 根据预测的起始位置和结束位置获取预测答案
                    true_answer = self.val_examples[idx * self.batch_size + i]['answer']  # 获取真实答案
                    print("pred_answer:" + pred_answer)  # 打印预测答案
                    print("true_answer:" + true_answer)  # 打印真实答案
                    true_answers.append(true_answer)  # 将真实答案添加到列表中
                    pred_answers.append(pred_answer)  # 将预测答案添加到列表中

        similarity = self.keywords_similarity(true_answers, pred_answers)  # 计算相似度得分
        return similarity  # 返回相似度得分

    def keywords_similarity(self, true_answers, pred_answers, top_k=5):
        vectorizer = TfidfVectorizer()  # 创建 TfidfVectorizer 实例
        corpus = true_answers + pred_answers  # 将真实答案和预测答案合并成一个列表
        X = vectorizer.fit_transform(corpus)  # 对合并后的列表进行向量化

        num_correct = 0  # 用于保存正确的相似度得分的计数器
        for i in range(len(true_answers)):  # 遍历每个真实答案
            true_answer = X[i]  # 获取真实答案的向量表示
            pred_answer = X[len(true_answers) + i]  # 获取预测答案的向量表示

            true_top_k_indices = true_answer.toarray()[0].argsort()[-top_k:][::-1]  # 获取真实答案中的前 k 个关键词的索引
            pred_top_k_indices = pred_answer.toarray()[0].argsort()[-top_k:][::-1]  # 获取预测答案中的前 k 个关键词的索引

            true_keywords = set([vectorizer.get_feature_names_out()[idx] for idx in true_top_k_indices])  # 获取真实答案中的关键词
            pred_keywords = set([vectorizer.get_feature_names_out()[idx] for idx in pred_top_k_indices])  # 获取预测答案中的关键词

            if len(true_keywords.intersection(pred_keywords)) > 0:  # 如果真实答案和预测答案有至少一个关键词相同
                num_correct += 1  # 计数器加一

        similarity = num_correct / len(true_answers) * 100  # 计算相似度得分
        return similarity  # 返回相似度得分

    def save_model(self):
        os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录
        print(f"Saving best model to {self.output_dir}")  # 打印正在保存模型的消息
        self.model.save_pretrained(self.output_dir)  # 保存模型的权重
        self.model.config.save_pretrained(self.output_dir)  # 保存模型的配置信息
