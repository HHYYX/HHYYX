# !/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/4/16 15:51 
# @Author : Snowball 
# @Email : Snowball96@163.com 
# @File : predict.py 
# @Software: PyCharm


import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering


def predict(model_path, tokenizer_path, context, question):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = BertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # Tokenize the context and the question
    inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

    # Decode the answer
    start = torch.argmax(start_logits)
    end = torch.argmax(end_logits)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokenizer.convert_tokens_to_string(tokens[start: end + 1])

    return answer


# Example usage
model_path = "../output/single_extrac"
tokenizer_path = "../model/chinese-bert-wwm"
question = "闵浦二桥的设计者是谁？"
context = "闵浦二桥，即沪闵路－沪杭公路越江工程，原名西渡大桥或奉浦二桥，是上海市的一座公路、轨道交通两用双层桥梁，为独塔双索斜拉桥形式；由上海市城市建设设计研究院所设计，武汉中铁大桥局集团承建。北接沪闵路，南连沪杭公路，是黄浦江上第九座大桥。上层（公路层）已于2010年5月21日通车。该桥北岸起自闵行区东川路以北，沿沪杭公路到奉贤区西渡镇西闸路以南落地，全长约5.8公里，距奉浦大桥约1.7公里。主桥全长436.55米，主跨长251.4米，轨道交通与公路叠合段长度为3.2公里。上层为双向四车道二级公路，宽度18米；下层为轨道交通5号线南延伸段，为电气化复线城市轨道交通。主桥设计为按300年一遇防洪标准，最高通航水位4.41米，最大通行3000吨级杂货船设计。"

answer = predict(model_path, tokenizer_path, context, question)
print("Answer:", answer)
