# config.py

from pathlib import Path

# 项目根目录
# __file__: 当前文件的路径，即 config.py 的路径
ROOT_DIR = Path(__file__).parent.parent.parent

# 模型路径
MODELS_DIR = ROOT_DIR / 'checkpoint'
# 日志路径
# LOGS_DIR = ROOT_DIR / 'logs'
# 服务器日志路径 autodl服务器
LOGS_DIR = Path('/root/tf-logs')
# 预训练目录
PRETRAIN_DIR = ROOT_DIR / 'pretrained'

# 原始数据文件
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
RAW_TRAIN_DATA_FILE = RAW_DATA_DIR / 'train.txt'
RAW_TEST_DATA_FILE = RAW_DATA_DIR / 'test.txt'
RAW_VALID_DATA_FILE = RAW_DATA_DIR / 'valid.txt'

# 预处理后路径
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

# 模型名称
BERT_MODEL_NAME = "google-bert/bert-base-chinese"
FINETUNED_MODEL_DIR = MODELS_DIR / 'best'

LABELS_FILE = MODELS_DIR / 'label.txt'

# 训练参数
SEQ_LEN = 128
BATCH_SIZE = 16 # 批大小 N
SAVE_STEP = 100

LEARNING_RATE = 1e-5 # 学习率 微调学习率就是要小一点
EPOCHS = 10 # 训练轮数
