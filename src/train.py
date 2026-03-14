"""
Train mô hình NCF (explicit + implicit) từ file .dat và lưu weights + CSV cho FastAPI.
Chạy từ thư mục gốc project: python src/train.py
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_SCRIPT_DIR, "model")
sys.path.insert(0, _MODEL_DIR)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.utils import Utils, EarlyStopping, cols_dict
from utils.model import NCF

DATA_DIR = os.path.join(_MODEL_DIR, "data")
WEIGHTS_DIR = os.path.join(_MODEL_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def load_data():
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::",
        names=cols_dict["ratings"],
        engine="python",
    )
    users = pd.read_csv(
        os.path.join(DATA_DIR, "users.dat"),
        sep="::",
        names=cols_dict["users"],
        engine="python",
    )
    items = pd.read_csv(
        os.path.join(DATA_DIR, "movies.dat"),
        sep="::",
        names=cols_dict["items"],
        encoding="latin-1",
        engine="python",
    )
    return ratings, users, items


def preprocess_explicit(ratings, users, items):
    """Chuẩn bị dữ liệu explicit. Trả về (ratings, users, items) đã xử lý."""
    users = Utils.one_hot_encode(users, ["occupation", "gender", "age"])
    items = Utils.multi_hot_encode(items, "genre")
    users = Utils.extract_category_avg_ratings(users, items, ratings)
    items = Utils.extract_year(items)
    users = Utils.move_column(users, ["gender_M", "gender_F"], 0)

    # Lưu users_exp cho API (một dòng mỗi user, còn user_id)
    users_exp = users.copy()
    users_exp.to_csv(os.path.join(DATA_DIR, "users_exp.csv"), index=False)

    users, items = Utils.extend_users_items(users, items, ratings)
    ratings = ratings.drop(["timestamp"], axis=1)
    users = users.drop(["user_id", "zip_code"], axis=1)
    items = items.drop(["movie_id", "title"], axis=1)
    ratings["rating"] = ratings["rating"].astype(float) / 5.0
    items["year"] = items["year"] / items["year"].max()
    users.iloc[:, -18:] = users.iloc[:, -18:] / users.iloc[:, -18:].max().max()
    return ratings, users, items


def preprocess_implicit(ratings, users, items):
    """Chuẩn bị dữ liệu implicit + negative sampling."""
    users = Utils.one_hot_encode(users, ["occupation", "gender", "age"])
    items = Utils.multi_hot_encode(items, "genre")
    users = Utils.extract_category_freq(users, items, ratings)
    items = Utils.extract_year(items)
    ratings = Utils.negative_sampling(ratings, items, num_negatives=20)
    users = Utils.move_column(users, ["gender_M", "gender_F"], 0)

    # Lưu users_imp cho API
    users_imp = users.copy()
    users_imp.to_csv(os.path.join(DATA_DIR, "users_imp.csv"), index=False)

    users, items = Utils.extend_users_items(users, items, ratings)
    items["year"] = items["year"].astype(float) / items["year"].max()
    ratings = ratings.drop(["timestamp"], axis=1)
    users = users.drop(["user_id", "zip_code"], axis=1)
    items = items.drop(["movie_id", "title"], axis=1)
    return ratings, users, items


def save_movies_csv(items_after_genre_year):
    """Lưu movies.csv cho API: movie_id + các cột feature (trước extend, có movie_id)."""
    df = items_after_genre_year.drop(columns=["title"], errors="ignore")
    df.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)


def split(users, items, ratings):
    u_train, u_test = train_test_split(users, test_size=0.2, shuffle=True, random_state=42)
    u_val, u_test = train_test_split(u_test, test_size=0.5, shuffle=True, random_state=42)
    i_train, i_test = train_test_split(items, test_size=0.2, shuffle=True, random_state=42)
    i_val, i_test = train_test_split(i_test, test_size=0.5, shuffle=True, random_state=42)
    r_train, r_test = train_test_split(ratings, test_size=0.2, shuffle=True, random_state=42)
    r_val, r_test = train_test_split(r_test, test_size=0.5, shuffle=True, random_state=42)

    return (
        u_train.values, u_val.values,
        i_train.values, i_val.values,
        r_train.values, r_val.values,
    )


def train_explicit(users_og, items_og):
    print("\n" + "=" * 60)
    print("1. HUẤN LUYỆN MÔ HÌNH EXPLICIT (dự đoán rating 1–5)")
    print("=" * 60)
    ratings, users_orig, items_orig = load_data()
    # Lưu movies.csv từ items (sau multi_hot + extract_year, trước extend)
    items_for_movies = Utils.multi_hot_encode(items_orig.copy(), "genre")
    items_for_movies = Utils.extract_year(items_for_movies)
    items_for_movies["year"] = items_for_movies["year"] / items_for_movies["year"].max()
    save_movies_csv(items_for_movies)

    ratings, users, items = preprocess_explicit(ratings, users_orig.copy(), items_orig.copy())
    u_tr, u_val, i_tr, i_val, r_tr, r_val = split(users, items, ratings)

    n_users = users_og["user_id"].max()
    n_items = items_og["movie_id"].max()
    model = NCF(
        num_users=n_users,
        num_items=n_items,
        user_dim=users.shape[1],
        item_dim=items.shape[1],
        num_factors=32,
        mode="explicit",
        criterion=torch.nn.MSELoss(),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        verbose=True,
        gpu=torch.cuda.is_available(),
    )
    early = EarlyStopping(patience=3, delta=0.0002, path=os.path.join(WEIGHTS_DIR, "explicit.pth"), verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode="min", factor=0.1, patience=0)

    model.fit(
        X=[r_tr[:, 0], r_tr[:, 1], u_tr, i_tr],
        y=r_tr[:, 2],
        X_val=[r_val[:, 0], r_val[:, 1], u_val, i_val],
        y_val=r_val[:, 2],
        epochs=12,
        batch_size=2048,
        early_stopping=early,
        scheduler=scheduler,
    )
    print("Đã lưu weights:", os.path.join(WEIGHTS_DIR, "explicit.pth"))


def train_implicit(users_og, items_og):
    print("\n" + "=" * 60)
    print("2. HUẤN LUYỆN MÔ HÌNH IMPLICIT (implicit feedback)")
    print("=" * 60)
    ratings, users_orig, items_orig = load_data()
    ratings, users, items = preprocess_implicit(ratings, users_orig.copy(), items_orig.copy())
    u_tr, u_val, i_tr, i_val, r_tr, r_val = split(users, items, ratings)

    n_users = users_og["user_id"].max()
    n_items = items_og["movie_id"].max()
    model = NCF(
        num_users=n_users,
        num_items=n_items,
        user_dim=users.shape[1],
        item_dim=items.shape[1],
        num_factors=32,
        mode="implicit",
        criterion=torch.nn.BCELoss(),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        verbose=True,
        gpu=torch.cuda.is_available(),
    )
    early = EarlyStopping(patience=2, delta=0.001, path=os.path.join(WEIGHTS_DIR, "implicit.pth"), verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode="min", factor=0.1, patience=0)

    model.fit(
        X=[r_tr[:, 0], r_tr[:, 1], u_tr, i_tr],
        y=r_tr[:, 2],
        X_val=[r_val[:, 0], r_val[:, 1], u_val, i_val],
        y_val=r_val[:, 2],
        epochs=10,
        batch_size=2048,
        k=10,
        early_stopping=early,
        scheduler=scheduler,
    )
    print("Đã lưu weights:", os.path.join(WEIGHTS_DIR, "implicit.pth"))


def main():
    print("Thư mục dữ liệu:", DATA_DIR)
    print("Thư mục lưu trọng số (weights):", WEIGHTS_DIR)
    ratings, users_og, items_og = load_data()
    print("Số dòng: ratings={}, users={}, items={}".format(len(ratings), len(users_og), len(items_og)))

    train_explicit(users_og, items_og)
    train_implicit(users_og, items_og)

    print("\nHoàn tất huấn luyện. Đã lưu weights và các file users_exp.csv, users_imp.csv, movies.csv trong", DATA_DIR)


if __name__ == "__main__":
    main()
