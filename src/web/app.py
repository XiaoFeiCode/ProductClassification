# 接口层级
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from runner.predict import Predictor

from configuration.config import *
from web.schemas import Title
from web.service import TitleService
from web.schemas import Category

app = FastAPI()
WEB_DIR = ROOT_DIR / "src" / "web"

# 跨域
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    # 允许的来源，可以直接写你的前端地址，或者用 *
    allow_origins=["*"],
    allow_credentials=True,
    # 必须包含 OPTIONS，这是解决 preflight 报错的关键！
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    # 允许所有请求头
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)

# 创建预测器
predictor = Predictor(model, tokenizer, device)

service = TitleService(predictor=predictor)


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/predict")
def predict(title: Title) -> Category:
    labels = service.predict(title.text)
    return Category(category=labels)


def server():
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
