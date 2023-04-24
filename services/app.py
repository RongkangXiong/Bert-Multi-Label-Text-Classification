# --coding:utf-8--
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from starlette.responses import HTMLResponse,  FileResponse
from transformers import BertTokenizer
from pybert.model.bert_for_multi_label import BertForMultiLable
from services.predict import data_predict
from services.config import config

app = FastAPI(title="goods same and similar probability")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

@app.on_event("startup")
async def load_model():
    global model, tokenizer, api_key
    try:
        print("begin load model...")
        # 从配置文件中读取API密钥
        try:
            if os.environ.get('USEDOCKER') == 'True':
                api_key = os.environ.get('APIKEY')
                print("Using Docker APIKEY")
            else:
                api_key = config['api_key']
                print("Using config APIKEY")
            
        except:
            api_key = config['api_key']
            print("Using config APIKEY")
        
        
        model = BertForMultiLable.from_pretrained(config['model_path'],
                                                  num_labels=config['num_labels']).to(device)
        tokenizer = BertTokenizer(config['vocab_path'], do_lower_case=config['do_lower_case'])
        print('load model success')
    except Exception as e:
        print("Load model failed:", e)
        raise HTTPException(status_code=500, detail="Load model failed")


# 定义请求数据模型
class Request(BaseModel):
    goods_a: str = "商品A名称"
    goods_b: str = "商品B名称"


# 定义响应数据模型
class Response(BaseModel):
    guid: str
    same_prob: float
    similar_prob: float


async def get_api_key(apikey: str = Header(None)):
    if apikey is None:
        raise HTTPException(status_code=400, detail="API key missing")
    elif apikey != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return apikey


@app.get("/", response_class=HTMLResponse)
async def index():
    """Basic HTML response."""
    return FileResponse(config["index"])


@app.post("/predict")
async def predict(input_text: Request, apikey: str = Depends(get_api_key)):
    global model, tokenizer
    str_a = input_text.goods_a
    str_b = input_text.goods_b

    examples, predict_result = data_predict(str_a=str_a,
                                            str_b=str_b,
                                            model=model,
                                            tokenizer=tokenizer,
                                            max_seq_len=config['max_seq_len'],
                                            example_type='predict',
                                            device=device)
    predictions = predict_result.detach().cpu().numpy().tolist()

    response = Response(
        guid=examples[0].guid,
        same_prob=predictions[0][0],
        similar_prob=predictions[0][1]
    )
    torch.cuda.empty_cache()
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
