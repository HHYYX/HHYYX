# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/17 8:25 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : data_format.py 
# @Software: PyCharm
import json


# 定义函数load_data，输入参数为content_file和answer_file
def load_data(content_file, answer_file):
    # 定义一个空列表content_data用于存储内容文件中的数据
    content_data = []
    # 打开内容文件，使用utf-8编码方式读取内容
    with open(content_file, 'r', encoding='utf-8') as f:
        # 逐行读取内容文件
        for line in f:
            try:
                # 尝试将每行内容转换为json格式并添加到content_data列表中
                content_data.append(json.loads(line.strip()))
            except json.decoder.JSONDecodeError:
                # 如果无法将该行内容转换为json格式，打印错误信息并继续处理下一行内容
                print(f"Error parsing line: {line}")
                continue

    # 打开答案文件，使用utf-8编码方式读取内容，并将内容存储到answer_data变量中
    with open(answer_file, 'r', encoding='utf-8') as f:
        answer_data = json.load(f)

    # 返回内容数据和答案数据
    return content_data, answer_data


# 定义函数preprocess_data，输入参数为content_data、answer_data和tokenizer
def preprocess_data(content_data, answer_data, tokenizer):
    # 定义一个空列表examples，用于存储处理后的数据
    examples = []
    # 遍历content_data列表中的每个item
    for item in content_data:
        # 获取每个item中的备选成语列表和内容列表
        candidates = item['candidates']
        content_list = item['content']

        # 遍历每个content，替换成语占位符为相应的备选成语，并生成相应的问题和编码
        for content in content_list:
            for idx, candidate in enumerate(candidates):
                # 替换成语占位符为相应的备选成语
                replaced_text = content.replace("#idiom000000#", candidate)
                # 生成问题，形如"在这个上下文中，哪个成语应该填充空白？选项 1: xxxxx"
                question = f"在这个上下文中，哪个成语应该填充空白？选项 {idx + 1}: {candidate}"
                # 使用tokenizer编码问题和替换后的内容，最大长度为384，并加入examples列表中
                encoded = tokenizer.encode_plus(question, replaced_text, return_tensors="pt", max_length=384,
                                                truncation=True, padding='max_length')
                examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "candidate": candidate,
                    "answer": answer_data.get(candidate, -1)
                })

    # 返回处理后的数据
    return examples


# 定义函数load_and_process_data，输入参数为train_content_path、train_answer_path、dev_content_path、dev_answer_path和tokenizer
def load_and_process_data(train_content_path, train_answer_path, dev_content_path, dev_answer_path, tokenizer):
    # 加载训练集和开发集的内容数据和答案数据
    train_content_data, train_answer_data = load_data(train_content_path, train_answer_path)
    dev_content_data, dev_answer_data = load_data(dev_content_path, dev_answer_path)

    # 处理训练集和开发集的数据
    train_data = preprocess_data(train_content_data, train_answer_data, tokenizer)
    dev_data = preprocess_data(dev_content_data, dev_answer_data, tokenizer)

    # 返回处理后的训练集和开发集数据
    return train_data, dev_data
