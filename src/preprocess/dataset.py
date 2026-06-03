from configuration.config import *

from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

# 获取数据集
def get_dataset(ds_type='train'):
    dataset = load_from_disk(str(PROCESSED_DATA_DIR / ds_type))
    return dataset

# 获取数据加载器
def get_dataloader(tokenizer, ds_type='train'):
    # 加载数据集
    dataset = load_from_disk(str(PROCESSED_DATA_DIR / ds_type))

    # 设置格式为 tensor
    dataset.set_format(type='torch')

    # 创建数据加载器
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    dataloader = get_dataloader(tokenizer)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        break
