# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 20:20 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : predict.py 
# @Software: PyCharm

import torch
from transformers import BertConfig, BertTokenizerFast, BertForMultipleChoice


def load_saved_model(config_path, model_path, num_choices):
    """
    从指定路径加载预训练的 BERT 模型，并根据选项数重新配置模型。

    Args:
        config_path (str): 预训练模型的配置文件路径。
        model_path (str): 预训练模型的权重文件路径。
        num_choices (int): 选项数。

    Returns:
        model: 已加载和重新配置的 BERT 模型。

    """
    # 从配置文件创建一个配置对象
    config = BertConfig.from_pretrained(config_path)

    # 根据选项数设置配置对象中的 num_choices 属性
    config.num_choices = num_choices

    # 从预训练模型文件中加载权重，使用新的配置对象进行重新配置
    model = BertForMultipleChoice.from_pretrained(model_path, config=config)

    # 返回已加载和重新配置的 BERT 模型
    return model



def predict(model, tokenizer, text, options, device):
    """
    对给定文本进行填空题预测，返回最有可能的选项。

    Args:
        model (BertForMultipleChoice): 预训练的 BERT 模型。
        tokenizer (BertTokenizer): 用于将输入转换为模型输入特征的 tokenizer。
        text (str): 包含填空的文本。
        options (list): 选项列表，包含可供选择填空的不同短语。
        device (str): 指定要在其上运行模型的设备。

    Returns:
        best_option (str): 最有可能的选项。

    """
    # 将模型和输入数据移动到指定的设备上，并将模型设置为评估模式
    model.to(device)
    model.eval()

    # 对每个选项生成一条问题和对应的模型输入
    examples = []
    for idx, option in enumerate(options):
        question = f"在这个上下文中，哪个成语应该填充空白？选项 {idx + 1}: {option}"
        encoded = tokenizer.encode_plus(question, text, return_tensors="pt", max_length=384, truncation=True,
                                        padding='max_length')
        examples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    # 将所有示例的输入合并为单个输入张量，以在模型中进行一次前向传递
    input_ids = torch.cat([example["input_ids"] for example in examples], dim=0).to(device)
    attention_mask = torch.cat([example["attention_mask"] for example in examples], dim=0).to(device)
    input_ids = input_ids.view(1, -1, input_ids.size(1))
    attention_mask = attention_mask.view(1, -1, attention_mask.size(1))

    # 在模型中进行前向传递并返回输出
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 从输出 logits 中获取最有可能的选项，并将其作为最终结果返回
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).item()
    best_option = options[preds]

    return best_option


if __name__ == "__main__":
    # 使用保存的模型进行预测
    config_path = "../output/multi_choice/best_model/config.json"
    model_path = "../output/multi_choice/best_model/pytorch_model.bin"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("../model/chinese-bert-wwm")

    options = ["吃喝玩乐", "听天任命", "暗送秋波", "相提而论", "以卵投石", "望其肩项", "成败利钝", "开门见山", "杀身成仁", "有目共见"]
    num_choices = len(options)
    model = load_saved_model(config_path, model_path, num_choices)

    # 示例：预测最佳选项
    text = "乔丹职业生涯前半段的敌人是伊赛亚-托马斯，后半段则是雷吉-米勒。但和托马斯相比，雷吉-米勒也就是一个很好的NBA球员，谈不上巨星，" \
           "他的球队步行者也只是一支实力还算不错的球队，和当年的活塞队无法#idiom000000#。乔丹，这位以冷静出名的超级明星实在没有必要和雷吉-米勒计较，" \
           "更何况对方的实力本来就比公牛差。" \

    best_option = predict(model, tokenizer, text, options, device)
    print(f"最佳选项：{best_option}")
