"""
Các endpoint API cho hệ gợi ý NCF.
- Lưu ý: Bản cuối dùng Streamlit thay vì FastAPI, file này chỉ dùng để test API.
"""

import os
os.sys.path.append(os.path.abspath('app/model/'))
abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/')

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.model import NCF, __model_version__
from utils.utils import Utils, cols_dict
from utils.requests import Request

app = FastAPI(title="API hệ gợi ý NCF", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trong thực tế, các file này sẽ được lưu trong cơ sở dữ liệu
users_exp = pd.read_csv(abs_path + 'data/users_exp.csv').values
users_imp = pd.read_csv(abs_path + 'data/users_imp.csv').values
movies = pd.read_csv(abs_path + 'data/movies.csv')
movies_og = pd.read_csv(abs_path + 'data/movies.dat', sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')
ratings = pd.read_csv(abs_path + 'data/ratings.dat', sep='::', names=cols_dict['ratings'], engine='python')

model_exp = NCF('explicit', gpu=False)
model_exp.load_weights(abs_path + 'weights/explicit.pth', eval=True)

model_imp = NCF('implicit', gpu=False)
model_imp.load_weights(abs_path + 'weights/implicit.pth', eval=True)

@app.get("/")
def root():
    return {
        "suc_khoe": "tốt",
        "phien_ban_mo_hinh": __model_version__,
        "mo_ta": "API hệ gợi ý NCF đang chạy bình thường"
    }

@app.post("/recommend/explicit")
def recommend_explicit(request: Request):
    return Utils.pipeline(
            request=request,
            model=model_exp,
            users=users_exp,
            movies=movies,
            movies_og=movies_og,
            ratings=ratings,
            weights=[model_exp.user_embedding_mlp.weight.data.cpu().numpy(), model_exp.user_embedding_mf.weight.data.cpu().numpy()],
            mode='explicit'
        )

@app.post("/recommend/implicit")
def recommend_implicit(request: Request):
    return Utils.pipeline(
        request=request,
        model=model_imp,
        users=users_imp,
        movies=movies,
        movies_og=movies_og,
        ratings=ratings,
        weights=[model_imp.user_embedding_mlp.weight.data.cpu().numpy(), model_imp.user_embedding_mf.weight.data.cpu().numpy()],
        mode='implicit'
    )

@app.post("/recommend/item-to-item", status_code=501)
def recommend_item_to_item():
    ...
    return {"message": "Tính năng này chưa được triển khai"}