import os
from argparse import ArgumentParser

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

if __name__ == '__main__':
    # 定义一个参数解析器
    parser = ArgumentParser(usage='python main.py [action]')

    # 添加具体参数
    parser.add_argument('action', choices=['preprocess', 'predict', 'train', 'evaluate', 'server'])

    # 解析参数
    arg = parser.parse_args()
    action = arg.action

    match action:
        case 'preprocess':
            from preprocess.preprocess import preprocess

            preprocess()
        case 'train':
            from runner.train import train

            train()
        case 'predict':
            from runner.predict import predict

            predict()
        case 'evaluate':
            from runner.evaluate import evaluate

            evaluate()
        case 'server':
            from web.app import server
            server()
