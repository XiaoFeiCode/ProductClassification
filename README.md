# 商品标题智能分类

这是一个面向电商运营场景的商品标题自动分类项目。运营人员上传或录入商品标题后，系统会使用微调后的中文 BERT 模型预测商品所属类别，可用于商品上架、类目推荐和基础审核流程。

## 功能

- 使用 `google-bert/bert-base-chinese` 做商品标题多分类。
- 支持原始 TSV 数据预处理。
- 支持训练、验证、测试集评估和单条/批量预测。
- 提供 FastAPI 接口 `/predict`。
- 提供挂载在 FastAPI 首页的商品上架运营工作台。

## 项目结构

```text
src/
  main.py                  # 命令入口
  configuration/config.py  # 路径和训练参数
  preprocess/              # 数据预处理和数据加载
  runner/                  # 训练、评估、预测
  web/                     # FastAPI 服务和前端页面
data/raw/                  # 原始训练、验证、测试数据
checkpoint/label.txt       # 类别名称
checkpoint/best/config.json # 已训练模型的配置元数据
```

## 环境

推荐使用 Python 3.10+。当前本地开发环境使用的是 Conda 环境 `product-classify`，Python 版本为 3.12。

```powershell
conda activate product-classify
pip install -r requirements.txt
```

项目默认使用 Hugging Face 镜像：

```text
HF_ENDPOINT=https://hf-mirror.com
```

如果你的环境可以直接访问 Hugging Face，也可以在启动前覆盖这个环境变量。

如果不激活环境，也可以直接调用环境里的解释器：

```powershell
D:\miniconda3\envs\product-classify\python.exe src\main.py --help
```

## 数据格式

原始数据是 TSV 格式，包含两列：

```text
label	text_a
家居	樱之歌蓝色之恋5件套日式釉下彩纯手绘家用餐具套装陶瓷器碗盘碗碟微波炉可用
```

数据文件位置：

- `data/raw/train.txt`
- `data/raw/valid.txt`
- `data/raw/test.txt`

## 运行

预处理数据：

```powershell
python src/main.py preprocess
```

训练模型：

```powershell
python src/main.py train
```

评估模型：

```powershell
python src/main.py evaluate
```

命令行预测：

```powershell
python src/main.py predict
```

启动 Web 服务：

```powershell
python src/main.py server
```

服务启动后，浏览器访问：

```text
http://127.0.0.1:8000/
```

接口请求地址：

```text
POST http://127.0.0.1:8000/predict
```

请求体示例：

```json
{
  "text": "华为双模5G全网通智能手机"
}
```

响应示例：

```json
{
  "category": "3C数码"
}
```

前端页面由 FastAPI 挂载在首页，源码在：

```text
src/web/index.html
```

## 模型文件说明

训练权重文件较大，不建议直接提交到普通 GitHub 仓库：

- `checkpoint/last/checkpoint.pt`
- `checkpoint/best/model.safetensors`

当前 `.gitignore` 已忽略这些大文件。上传 GitHub 后，如果需要提供可直接推理的模型，可以选择：

- 使用 Git LFS 管理模型权重。
- 将模型上传到 Hugging Face Hub 或其他网盘，并在 README 中放下载地址。
- 只上传代码和数据，让使用者自行运行训练命令生成模型。

## 类别

当前模型支持 30 个商品类别，类别清单保存在：

```text
checkpoint/label.txt
```
