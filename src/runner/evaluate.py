import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from preprocess.dataset import get_dataset
from runner.train import Trainer
from configuration.config import *


# 验证流程
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 2. 加载微调后的模型
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / 'best')

    # 3. 定义数据集和整理函数
    test_dataset = get_dataset(ds_type='test')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')

    # 3.1 评估函数，根据需求定义
    def compute_metrics(preds, labels):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'acc': acc, 'f1': f1}

    # 定义训练器
    trainer = Trainer(model, valid_dataset=test_dataset, collate_fn=collate_fn, compute_metrics=compute_metrics,
                      device=device)

    # 评估
    metrics = trainer.evaluate()
    metrics_str = '|'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    print(f'Metrics: {metrics_str}')

if __name__ == '__main__':
    evaluate()
