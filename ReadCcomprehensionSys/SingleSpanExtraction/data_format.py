# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 15:43 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : data_format.py 
# @Software: PyCharm

import json


def is_prediction_correct(pred_answer, true_answer):
    # 从索引 0 开始逐个字符地去匹配预测句子
    for i in range(len(pred_answer)):
        substring_to_check = pred_answer[i:]
        if substring_to_check in true_answer:
            return True

    # 如果没有找到匹配的子句，则认为预测是错误的
    return False


# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 从指定路径读取 JSON 文件

    examples = []  # 定义一个空列表，用于存储每个例子

    # 遍历每个段落，提取上下文、问题和答案，并将它们添加到例子列表中
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']  # 获取上下文文本
            for qa in paragraph['qas']:
                question = qa['question']  # 获取问题文本
                answer_start = qa['answers'][0]['answer_start']  # 获取答案开始字符位置
                answer_text = qa['answers'][0]['text']  # 获取答案文本
                examples.append({
                    'context': context,
                    'question': question,
                    'answer_start': answer_start,
                    'answer': answer_text
                })  # 将上下文、问题和答案添加到例子列表中

    return examples  # 返回例子列表
