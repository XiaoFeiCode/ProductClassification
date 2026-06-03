import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from configuration.config import *
from datasets import load_dataset

from transformers import AutoTokenizer


def preprocess():
    # 1. 读取原始数据
    dataset_dict = load_dataset('csv', data_files={
        'train': str(RAW_TRAIN_DATA_FILE),
        'valid': str(RAW_VALID_DATA_FILE),
        'test': str(RAW_TEST_DATA_FILE)
    },
                                # 分隔符
                                delimiter='\t')

    # 2. 过滤掉无效数据
    dataset_dict = dataset_dict.filter(lambda x: x['label'] is not None and x['text_a'] is not None)

    # 3. 将label转换成一个ClassLabel 相当于转成id了
    dataset_dict = dataset_dict.class_encode_column('label')
    # 3.1 获取它自动生成的标签列表，用来保存
    all_labels = dataset_dict['train'].features['label'].names
    # 3.2 保存类别 id -> label 映射关系
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))

    # 4. 对文本进行分词
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize(batch):
        inputs = tokenizer(
            batch['text_a'],
            truncation=True,
            # 填充可以等到批次的填充，这样会快很多
        )
        inputs['labels'] = batch['label']
        return inputs

    dataset_dict = dataset_dict.map(tokenize, batched=True, remove_columns=['label', 'text_a'])
    print(dataset_dict['train'][:3])

    # 5. 保存数据集
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    preprocess()
