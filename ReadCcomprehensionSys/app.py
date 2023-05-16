from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, BertForMultipleChoice

app = Flask(__name__)


def predict(model_path, tokenizer_path, context, question):
    # 确定计算设备是CPU还是GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的BERT模型以进行问答
    model = BertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 加载预训练的BERT分词器
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # 对上下文和问题进行分词
    inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

    # 解码预测的答案
    start = torch.argmax(start_logits)
    end = torch.argmax(end_logits)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokenizer.convert_tokens_to_string(tokens[start: end + 1])

    # 返回预测的答案
    return answer


def predict_multiple_choice(model_path, tokenizer_path, context, options):
    # 确定计算设备是CPU还是GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的BERT多项选择模型
    model = BertForMultipleChoice.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 加载预训练的BERT分词器
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    examples = []
    # 对每个选项进行编码
    for idx, option in enumerate(options):
        question = f"在这个上下文中，哪个成语应该填充空白？选项 {idx + 1}: {option}"
        encoded = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=384, truncation=True,
                                        padding='max_length')
        examples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    # 组合编码后的选项张量并添加批次维度
    input_ids = torch.cat([example["input_ids"] for example in examples], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.cat([example["attention_mask"] for example in examples], dim=0).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 获取预测结果中的最高分数对应的选项
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).item()
    best_option = options[preds]

    # 返回预测结果
    return best_option


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_question():
    data = request.get_json(force=True)
    context = data['context']
    question = data['question']
    question_type = data['question_type']
    options_str = data['options']  # 获取选项字符串
    options = options_str[0].split('，')  # 将字符串拆分为单独的选项

    model_path = "./output"
    tokenizer_path = "./model/chinese-bert-wwm"

    if question_type == "multiple_choice":
        # 调用多选式阅读理解的预测函数
        answer = predict_multiple_choice(model_path, tokenizer_path, context, options)
    else:
        # 调用单跨抽取式阅读理解的预测函数
        answer = predict(model_path, tokenizer_path, context, question)

    response = {
        "answer": answer
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
