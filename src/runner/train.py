import os
import time
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from preprocess.dataset import get_dataset
from torch.optim import Adam
from torch.cuda.amp import GradScaler

from configuration.config import *


# 训练配置类
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-5
    save_step: int = 100
    save_dir: Path = MODELS_DIR
    log_dir: Path = LOGS_DIR
    early_stop_metric: str = 'loss'  # 早停指标
    early_stop_patience: int = 5  # 多少次不降低或者不变化才停止
    # 是否开启amp功能
    amp: bool = True


class Trainer:
    # 初始化方法
    def __init__(self, model, valid_dataset, collate_fn, compute_metrics, device,
                 train_dataset=None, train_config=TrainConfig):
        # 训练参数配置
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        # 评价指标
        self.compute_metrics = compute_metrics
        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=self.train_config.lr)

        # 全局的迭代次数（运行的step数）
        self.global_step = 1
        # TensorBoard写入器
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime('%Y-%m-%d-%H-%M-%S')))
        # 全局最佳评估得分
        self.early_stop_score = -float('inf')
        # 容忍度计数器
        self.tolerance_counter = 0

        # 缩放器
        self.scaler = GradScaler(enabled=self.train_config.amp)

        # 检查点文件路径
        self.checkpoint_file = Path(self.train_config.save_dir) / 'last' / 'checkpoint.pt'

    # 定义内部方法，获取数据加载器
    def _get_dataloader(self, dataset):
        # 设置格式为 tensor
        dataset.set_format(type='torch')

        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True,
                                collate_fn=self.collate_fn)
        return dataloader

    # 核心方法
    def train(self):
        # 加载检查点
        self._load_checkpoint()

        # 获取当前的训练集加载器
        train_dataloader = self._get_dataloader(self.train_dataset)
        # 训练模式
        self.model.train()

        # 双重for循环，外层遍历所有的epoch
        for epoch in range(self.train_config.epochs):
            # 内层遍历所有的batch
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', position=0):
                # 调用一步（step）
                this_loss = self._train_one_step(batch)

                # 判断如果达到了save_step, 做计算损失，保存模型
                if self.global_step % self.train_config.save_step == 0:
                    # 计算损失
                    tqdm.write(f'Epoch {epoch + 1}, Step {self.global_step}, Loss: {this_loss:.4f}')
                    self.writer.add_scalar('Loss', this_loss, self.global_step)

                    # 新增验证指标
                    valid_metrics = self.evaluate()
                    valid_metrics_str = '|'.join([f'{k}: {v:.4f}' for k, v in valid_metrics.items()])
                    tqdm.write(f'Valid Metrics: {valid_metrics_str}')

                    # 早停判断处理
                    if self._should_stop(valid_metrics):
                        tqdm.write("早停！")
                        break

                    # # 判断如果这个loss比最小的loss小，保存模型
                    # if this_loss < self.min_loss:
                    #     self.min_loss = this_loss
                    #     # 保存模型
                    #     self.model.save_pretrained(self.train_config.save_dir)
                    #     tqdm.write("保存模型！")

                    # 检查点机制
                    self._save_checkpoint()

                self.global_step += 1

    # 一步训练 一批数据 内部函数加一个_
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.train_config.amp):
            # 前向传笔+计算损失
            outputs = self.model(**inputs)
            loss_value = outputs.loss

        # 反向传播+梯度更新
        self.scaler.scale(loss_value).backward()
        self.scaler.step(self.optimizer)
        # 更新缩放器
        self.scaler.update()
        # 清零梯度
        self.optimizer.zero_grad()

        return loss_value.item()

    # 核心验证方法，返回一个字典，记录不同的评价指标{'loss': 0.5, 'acc': 0.8, 'f1': 0.9}
    def evaluate(self):
        # 加载验证集
        valid_dataloader = self._get_dataloader(self.valid_dataset)
        # 验证模式
        self.model.eval()

        total_loss = 0.0
        all_labels = []  # 真实标签
        all_preds = []  # 预测标签

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc='Evaluating', position=1):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss_value = outputs.loss

                # 计算损失
                total_loss += loss_value.item()
                # 获取预测标签
                logits = outputs.logits
                # 每条数据多分类的维度取每行最大的值
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(inputs['labels'].tolist())
            # 遍历完毕，计算平均损失和其他指标
            loss = total_loss / len(valid_dataloader)
            # 返回一组指标的字典
            metrics = self.compute_metrics(all_preds, all_labels)
            # 字典解包
            return {'loss': loss, **metrics}

    def _should_stop(self, valid_metrics):
        # 提取评价指标值
        metric_value = valid_metrics[self.train_config.early_stop_metric]
        # 转换成评估评分
        # 越大的loss,加负号之后，score就变的越小
        score = -metric_value if self.train_config.early_stop_metric == 'loss' else metric_value
        if score > self.early_stop_score:
            self.early_stop_score = score
            self.tolerance_counter = 0
            tqdm.write('保存模型!')
            self.model.save_pretrained(str(self.train_config.save_dir / 'best'))
        else:
            self.tolerance_counter += 1
            # 判断是否达到早停的次数
            if self.tolerance_counter >= self.train_config.early_stop_patience:
                return True
            else:
                return False

    # 保存检查点函数
    def _save_checkpoint(self):
        # 定义字典，用于保存检查点
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': self.global_step,
            # 早停相关参数
            'early_stop_score': self.early_stop_score,
            'tolerance_counter': self.tolerance_counter
        }

        torch.save(checkpoint, self.checkpoint_file)

    # 加载检查点函数
    def _load_checkpoint(self):
        # 如果存在就加载
        if self.checkpoint_file.exists():
            tqdm.write("加载检查点...")
            # 加载检查点字典
            checkpoint = torch.load(self.checkpoint_file)
            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # 加载全局步数
            self.global_step = checkpoint['step']
            # 加载早停相关参数
            self.early_stop_score = checkpoint['early_stop_score']
            self.tolerance_counter = checkpoint['tolerance_counter']
        else:
            tqdm.write("检查点不存在！")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 1.1 加载label
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        all_labels = f.read().split('\n')

    # id -> label
    id2label = {i: label for i, label in enumerate(all_labels)}
    label2id = {label: i for i, label in enumerate(all_labels)}

    # 2. 加载模型 带任务头可以自动加载id和label的对应关系
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME,
                                                               num_labels=len(all_labels),
                                                               id2label=id2label,
                                                               label2id=label2id)

    # 3. 定义数据集和整理函数
    train_dataset = get_dataset(ds_type='train')
    valid_dataset = get_dataset(ds_type='valid')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')

    # 3.1 评估函数，根据需求定义
    def compute_metrics(preds, labels):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'acc': acc, 'f1': f1}

    # 4. 训练参数配置
    train_config = TrainConfig(batch_size=BATCH_SIZE, lr=LEARNING_RATE, save_step=SAVE_STEP, log_dir=LOGS_DIR)
    # 5. 创建训练器
    trainer = Trainer(
        model=model,
        valid_dataset=valid_dataset,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        device=device,
        train_dataset=train_dataset,
        train_config=train_config)
    # 6. 开始训练
    trainer.train()


if __name__ == '__main__':
    train()
