# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 21:21 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : BertWithAttention.py
# @Software: PyCharm

from transformers import BertModel, BertForQuestionAnswering, BertTokenizerFast
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
import torch


class BertForQuestionAnsweringWithAttention(BertForQuestionAnswering):
    # 定义一个继承自BertForQuestionAnswering的类
    def __init__(self, config):
        # 构造函数
        super().__init__(config)
        # 调用父类的构造函数
        self.bert = BertModel(config, add_pooling_layer=False)
        # 初始化BertModel

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # 定义前向传播函数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置返回值字典

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,  # 设置output_attentions=True以获取注意力
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用BertModel进行前向传播，并获得输出

        sequence_output = outputs[0]
        # 获取输出的序列输出
        pooled_output = self.qa_outputs(sequence_output)
        # 将序列输出传入qa_outputs层，获得池化输出
        start_logits, end_logits = pooled_output.split(1, dim=-1)
        # 将池化输出分裂成开始位置和结束位置的概率分布
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # 如果有提供开始和结束位置的标签
            loss_fct = CrossEntropyLoss()
            # 定义交叉熵损失函数
            start_loss = loss_fct(start_logits, start_positions)
            # 计算开始位置的损失
            end_loss = loss_fct(end_logits, end_positions)
            # 计算结束位置的损失
            total_loss = (start_loss + end_loss) / 2
            # 计算总损失
            return (total_loss,) + outputs[2:] if return_dict else (total_loss,) + outputs[1:]
            # 返回总损失和其他输出

        if return_dict:
            return {"start_logits": start_logits, "end_logits": end_logits, "attentions": outputs.attentions}
        else:
            return start_logits, end_logits, outputs.attentions
        # 返回开始位置和结束位置的概率分布以及注意力信息（如果指定输出字典，则使用字典形式返回结果）


def plot_attention_barchart(attentions_matrix, tokens, question_length, answer_start, answer_end):
    # 定义一个函数来绘制注意力权重柱状图，输入参数包括注意力权重矩阵，标记序列，问题长度，答案起始位置和结束位置

    # Average the attention weights across all heads
    # 对所有头的注意力权重进行平均
    attentions_avg = attentions_matrix.mean(axis=(0, 1))

    # Normalize the averaged attention weights
    # 将平均的注意力权重进行归一化
    attentions_avg = attentions_avg - attentions_avg.min()
    attentions_avg = attentions_avg / attentions_avg.max()

    # Separate tokens and weights for the question and context
    # 分离问题和上下文的标记和权重
    question_tokens = tokens[:question_length]
    context_tokens = tokens[question_length:]
    question_attentions = attentions_avg[:question_length]
    context_attentions = attentions_avg[question_length:]

    # Filter out special tokens
    # 过滤掉特殊的标记，如[PAD]和[UNK]
    filtered_tokens = [(t, a) for t, a in zip(context_tokens, context_attentions) if t not in ["[PAD]", "[UNK]"]]

    fig, ax = plt.subplots(figsize=(20, 5))

    # Plot the context attentions
    # 绘制上下文的注意力权重柱状图
    ax.bar([t[0] for t in filtered_tokens], [t[1] for t in filtered_tokens], color='orange', label='Context', width=0.5)

    # Highlight the answer tokens
    # 突出显示答案标记
    for j in range(answer_start - question_length, answer_end - question_length + 1):
        ax.get_children()[j].set_color('lightgreen')

    ax.set_xticks(range(len(filtered_tokens)))
    ax.set_xticklabels([t[0] for t in filtered_tokens], rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_title('Attention Weights for Context Tokens')
    plt.show()
    # 设置横坐标刻度，标签和字体样式，并绘制柱状图


def predict_with_attention(model_path, tokenizer_path, context, question):
    # 定义一个函数用于预测答案，返回注意力权重、输入标记ID

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型并将其移动到设备上
    model = BertForQuestionAnsweringWithAttention.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # 对输入的问题和上下文进行分词
    inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 从input_ids中获取标记
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 进行预测并获取注意力权重
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits, end_logits, attentions = outputs["start_logits"], outputs["end_logits"], outputs["attentions"]

    # 获取预测答案的起始和结束索引
    start = torch.argmax(start_logits)
    end = torch.argmax(end_logits)

    # 将预测的答案标记转换回文本
    answer = tokenizer.convert_tokens_to_string(tokens[start: end + 1])

    # 返回预测的答案，注意力权重和input_ids
    return answer, attentions, input_ids


if __name__ == "__main__":
    # Example usage
    model_path = "../output"
    tokenizer_path = "../model/chinese-bert-wwm"
    question = "闵浦二桥的设计者是谁？"
    context = "闵浦二桥，即沪闵路－沪杭公路越江工程，原名西渡大桥或奉浦二桥，是上海市的一座公路、轨道交通两用双层桥梁，为独塔双索斜拉桥形式；由上海市城市建设设计研究院所设计，武汉中铁大桥局集团承建。北接沪闵路，南连沪杭公路，是黄浦江上第九座大桥。上层（公路层）已于2010年5月21日通车。该桥北岸起自闵行区东川路以北，沿沪杭公路到奉贤区西渡镇西闸路以南落地，全长约5.8公里，距奉浦大桥约1.7公里。主桥全长436.55米，主跨长251.4米，轨道交通与公路叠合段长度为3.2公里。上层为双向四车道二级公路，宽度18米；下层为轨道交通5号线南延伸段，为电气化复线城市轨道交通。主桥设计为按300年一遇防洪标准，最高通航水位4.41米，最大通行3000吨级杂货船设计。"

    # Get answer and attentions from the predict function
    answer, attentions, input_ids = predict_with_attention(model_path, tokenizer_path, context, question)
    print("Answer:", answer)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)  # 创建 tokenizer 实例
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Get the answer start and end positions in the token list
    answer_start = tokens.index("[CLS]") + len(tokenizer.tokenize(question)) + 1
    answer_end = answer_start + len(tokenizer.tokenize(answer)) - 1

    # Plot the attention heatmap
    first_layer_attention = attentions[-1][0].detach().cpu().numpy()

    plot_attention_barchart(first_layer_attention, tokens, len(tokenizer.tokenize(question)), answer_start, answer_end)
