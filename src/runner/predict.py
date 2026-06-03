import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configuration.config import *


# 预测器类
class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str | list):
        # 统一数据格式，如果是字符串，则转为列表
        is_str = isinstance(text, str)
        if is_str:
            text = [text]
        # 1. 模型编码，得到输入
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 2. 模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 3. 获取预测结果
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).tolist()
        labels = [self.model.config.id2label[pred_id] for pred_id in preds]

        # 字符串进来是一个一维的，一条数据，列表进来是一个二维的，多条数据
        if is_str:
            return labels[0]
        else:
            return labels


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)

    # 创建预测器
    predictor = Predictor(model, tokenizer, device)

    # 预测
    text = '好奇心钻装纸尿裤L40片9-14kg'
    result1 = predictor.predict(text)
    print(result1)

    texts = ['基地玉米.', '潘婷丝质顺滑洗发露750ml']
    result2 = predictor.predict(texts)
    print(result2)


if __name__ == '__main__':
    predict()
