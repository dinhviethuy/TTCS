import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from typing import Literal, Union
import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

class Utils:
    @staticmethod
    def extract_year(items_df: pd.DataFrame) -> pd.DataFrame:
      items_df['year'] = items_df['title'].str.extract(r'\((\d{4})\)').astype(int)

      return items_df

    @staticmethod
    def extract_category_avg_ratings(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame, k=0.6) -> pd.DataFrame:
      # Tạo bảng features
      features_df = users_df.copy()

      # Hàm exponential penalty
      def exp_penalty(n, k=0.6):
          return 1 / np.exp(k * n)

      # Lặp qua từng category trong items_df (loại trừ các cột không phải category)
      for category in items_df.columns[2:]:
        # Lấy ID của item trong category hiện tại
        category_items = items_df[items_df[category] == 1]['movie_id']

        # Lọc ratings_df để chỉ bao gồm ratings cho items trong category hiện tại
        category_ratings = ratings_df[ratings_df['movie_id'].isin(category_items)]

        # Group by user_id và tính toán average rating và số lượng ratings cho category hiện tại
        user_stats = category_ratings.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
        user_stats.columns = ['user_id', f'user_avg_rating_{category}', f'count_rating_{category}']

        # Merge user stats vào features_df
        features_df = pd.merge(features_df, user_stats, on='user_id', how='left')

        # Áp dụng exponential penalty và tính toán average rating penalized cho mỗi user
        features_df[f'user_avg_rating_{category}'] = (
            (1 - exp_penalty(features_df[f'count_rating_{category}'], k)) * features_df[f'user_avg_rating_{category}']
        )

        # Điền giá trị missing với 0
        features_df[f'user_avg_rating_{category}'] = features_df[f'user_avg_rating_{category}'].fillna(0)

      # Chọn chỉ các cột liên quan
      cols = features_df.columns[:32].tolist() + [col for col in features_df.columns if col.startswith('user_avg_rating_')]
      result_df = features_df[cols]

      return result_df

    @staticmethod
    def extract_category_freq(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
      # Copy users_df để tránh sửa đổi dataframe gốc
      freq_df = users_df.copy()

      # Lấy tổng số interactions cho mỗi user
      total_interactions = ratings_df.groupby('user_id').size().reset_index(name='total_interactions')
      freq_df = pd.merge(freq_df, total_interactions, on='user_id', how='left').fillna(0)
      
      # Lặp qua từng category trong items_df (giả sử categories là từ cột thứ 3 trở đi)
      for category in items_df.columns[2:]:
        # Lấy movie_ids thuộc category hiện tại
        movie_ids_in_category = items_df[items_df[category] == 1]['movie_id']
        
        # Đếm interactions trong category hiện tại cho mỗi user
        category_interactions = ratings_df[ratings_df['movie_id'].isin(movie_ids_in_category)].groupby('user_id').size().reset_index(name=f'{category}_count')
        
        # Merge category_interactions vào freq_df
        freq_df = pd.merge(freq_df, category_interactions, on='user_id', how='left').fillna(0)
        
        # Tính tần suất interactions cho category hiện tại
        freq_df[f'freq_{category}'] = freq_df[f'{category}_count'] / freq_df['total_interactions']
        
        # Xóa cột tạm thời để tránh trùng lặp
        freq_df.drop(columns=[f'{category}_count'], inplace=True)
      
      return freq_df.fillna(0).drop(columns=['total_interactions'])

    @staticmethod
    def extend_users_items(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
      # Kéo dài users_df
      users_df = pd.merge(users_df, ratings_df[['user_id']], on='user_id', how='right')

      # Kéo dài items_df
      items_df = pd.merge(items_df, ratings_df[['movie_id']], on='movie_id', how='right')
      
      return users_df, items_df
    
    @staticmethod
    def multi_hot_encode(df: pd.DataFrame, col: str, delimiter='|') -> pd.DataFrame:
      df_ = df.copy(deep=True)

      # Thay thế Children's bằng Children để trùng với các genres khác
      df_[col] = df_[col].str.replace("Children's", 'Children') if col == 'genre' else df_[col]

      # tách genres
      df_[col] = df_[col].str.split(delimiter)

      # Tạo bảng pivot
      pivot_df = df_.explode(col).pivot_table(index='movie_id', columns=col, aggfunc='size', fill_value=0).reset_index()

      # Merge bảng pivot với dataframe gốc trên 'movie_id'
      result = pd.merge(df, pivot_df, on='movie_id', how='left')
      
      return result.drop(columns=[col])
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
      return pd.get_dummies(df, columns=cols) * 1
    
    @staticmethod
    def move_column(df: pd.DataFrame, col: list[str], pos: int) -> pd.DataFrame:
      cols = df.columns.tolist()
      for i in reversed(col):
        cols.insert(pos, cols.pop(cols.index(i)))
      return df[cols]

    @staticmethod
    def preprocess_user(user: dict, num_items: int, users: np.ndarray, weights: list[np.ndarray]=None, topk: int=3, verbose=False) -> tuple[torch.IntTensor, torch.FloatTensor, Union[list[np.ndarray], None], Union[np.ndarray, None]]:
      if 'age' not in user or not user['age']:
        user_ = users[user['id'] - 1]
        user_ = np.insert(user_, 0, user['id'])
        print(f"User id: {user['id']} top {topk} genres: {np.array(genre)[np.argsort(user_[-18:])[-topk:][::-1]]}") if verbose else None
        user_ = np.tile(user_, (num_items, 1))
        return torch.IntTensor(user_[:, 0]), torch.FloatTensor(user_[:, 1:]), None, np.array(genre)[np.argsort(user_[0, -18:])[-topk:][::-1]]

      user_ = np.zeros(31, dtype=float)

      user_[0] = user['id']

      user_[1 if user['gender'] == 'M' else 2] = 1

      user_[3 + occupation.index(user['occupation'])] = 1

      # ánh xạ age đến bins
      user['age'] = 1 if user['age'] < 18 else 18 if user['age'] < 25 else 25 if user['age'] < 35 else 35 if user['age'] < 45 else 45 if user['age'] < 56 else 56

      user_[3 + len(occupation) + age.index(user['age'])] = 1

      avg_ratings = np.zeros(len(genre), dtype=float) # 18 genres

      for genre_ in user['genres']:
        avg_ratings[genre.index(genre_)] = 1.0

      user_ = np.concatenate((user_, avg_ratings))

      # Lấy top 10 users ids của users với similar intrests (cosine similarity)
      similar_users_ids = cosine_similarity(user_[1:].reshape(1, -1), users).argsort()[0][-10:]

      # Lấy mean embeddings của top 10 users tương tự
      mlp_weights = weights[0][similar_users_ids].mean(axis=0)
      mf_weights = weights[1][similar_users_ids].mean(axis=0)

      user_ = np.tile(user_, (num_items, 1))
      return torch.IntTensor(user_[:, 0]), torch.FloatTensor(user_[:, 1:]), [mlp_weights, mf_weights], None
    
    @staticmethod
    def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
      # multi hot encode genres
      items_ = Utils.multi_hot_encode(items, 'genre')
      items_ = Utils.extract_year(items_)
      items_['year'] = items_['year'] / items_['year'].max()
      items_ = items_.drop(['title'], axis=1)
      
      return items_

    @staticmethod
    def remove_missing_values(ratings: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame]:
      # Lấy item ids với release date missing
      nan_item_ids = items[items[['release_date']].isna().any(axis=1)]['item_id']

      # Xóa movies với release date missing
      items.dropna(subset=['release_date'], inplace=True)

      # xóa ratings của items missing
      ratings = ratings[~ratings['item_id'].isin(nan_item_ids)]

      return ratings, items

    @staticmethod
    def negative_sampling(ratings: pd.DataFrame, items: pd.DataFrame, num_negatives: int) -> pd.DataFrame:
      # Lấy tất cả movie ids
      all_items = items['movie_id'].values

      negative_samples = []
      for user_id in ratings['user_id'].unique():
        # Movie ids mà user đã tương tác
        pos_items = ratings[ratings['user_id'] == user_id]['movie_id'].values

        # Movie ids mà user chưa tương tác
        unrated_items = np.setdiff1d(all_items, pos_items)

        # Lấy negative items
        neg_items = np.random.choice(unrated_items, size=num_negatives, replace=False)

        # Tạo negative samples
        for item_id in neg_items:
          negative_samples.append([user_id, item_id, 0])

      negative_samples = pd.DataFrame(negative_samples, columns=['user_id', 'movie_id', 'rating'])

      ratings['rating'] = [1] * ratings.shape[0]

      return pd.concat([ratings, negative_samples], ignore_index=True)
    
    def ndcg_hit_ratio(y_preds, X_test_users, y_true, k=10) -> tuple[float]:
      unique_users = np.unique(X_test_users, axis=0)

      hits = 0
      total_users = len(unique_users)

      y_preds_padded = []
      y_true_padded = []
      for user in unique_users:
        # Lấy indices của user
        user_indices = np.where((X_test_users == user).all(axis=1))[0]
        # Lấy predictions cho user
        user_preds = y_preds[user_indices][:k].flatten()
        # Lấy true ratings cho user
        user_true = y_true[user_indices][:k].flatten()

        # Tính số lượng hits
        if np.any(user_true == 1):
          hits += 1

        # Pad sublists để có k elements
        if len(user_preds) < k:
          user_preds = np.pad(user_preds, (0, k - len(user_preds)), mode='constant', constant_values=-1e10)
        if len(user_true) < k:
          user_true = np.pad(user_true, (0, k - len(user_true)), mode='constant', constant_values=0)

        y_preds_padded.append(user_preds)
        y_true_padded.append(user_true)

      # Tính NDCG
      ndcg = ndcg_score(y_true_padded, y_preds_padded, k=k)
      # Tính hit ratio
      hit_ratio = hits / total_users
      return ndcg, hit_ratio
    
    @staticmethod
    def pipeline(request: any, model: nn.Module, weights: list[np.ndarray], users: np.ndarray, movies: pd.DataFrame, movies_og: pd.DataFrame, ratings: pd.DataFrame, mode: str) -> tuple[list[dict], Union[np.ndarray, None]]:
      num_items = 300 # Số lượng items để lấy
      request = request if isinstance(request, dict) else request.model_dump()

        # preprocess user cũ
      user_id, user, weights, top_n_genres = Utils.preprocess_user(
        user=request,
        num_items=num_items,
        users=users,
        weights=weights
      )
      user_id, user = user_id.to(model.device), user.to(model.device)

      movies = Utils.retrieve(
        movies=movies,
        user=user.detach().cpu().numpy(),
        num_genres=len(request['genres']) if request['genres'] else 3,
        k=num_items,
        random_state=0
      )

      movie_ids, movies = Utils.filter(
        movies=movies,
        ratings=ratings,
        user_id=request['id']
      )
      movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

      y_pred = model(
        user_id[:len(movies)],
        movie_ids,
        user[:len(movies)],
        movies,
        weights
      ).cpu().detach().numpy()

      movies_retrieved = movies_og[movies_og['movie_id'].isin(movie_ids.cpu().numpy())].sort_values(by='movie_id', key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True))

      return Utils.order(y_pred, movies_retrieved, mode, top_k=request['top_k']).to_dict(orient='records'), top_n_genres
    
    @staticmethod
    def retrieve(movies: pd.DataFrame, user: np.ndarray, k: int, num_genres: int=3, random_state: int=42) -> pd.DataFrame:
      num_movies_per_genre = k // (num_genres + 1)
      most_popular_genres = ['Drama', 'Comedy', 'Action'] # Trong trường hợp thực tế, điều này sẽ thay đổi tuần tùy theo genres phổ biến

      # Lấy 3 genres phổ biến nhất mà user thích
      top_n_genres = np.array(genre)[np.argsort(user[0, -18:])[-num_genres:][::-1]]
      
      movies_ = []
      # Lấy movies cho mỗi genre ngẫu nhiên, vì chúng ta không có ratings của movies để sắp xếp
      for g in top_n_genres:
        m = movies[movies[g] == 1]
        if m.shape[0] < num_movies_per_genre: # Kiểm tra nếu có đủ movies cho genre
            movies_.append(m)
            continue
        movies_.append(m.sample(num_movies_per_genre, random_state=random_state))                
      
      # Lấy movies cho các genres phổ biến nhất
      for g in most_popular_genres:
        m = movies[movies[g] == 1]
        if m.shape[0] < num_movies_per_genre//3: # Kiểm tra nếu có đủ movies cho genre
          movies_.append(m)
          continue
        movies_.append(movies[movies[g] == 1].sample(num_movies_per_genre//3, random_state=random_state))

      return pd.concat(movies_, ignore_index=True)
    
    @staticmethod
    def filter(movies: pd.DataFrame, ratings: pd.DataFrame, user_id: int) -> tuple[torch.IntTensor, torch.FloatTensor]:
      # Lấy movie ids mà user chưa tương tác
      user_movies = ratings[ratings['user_id'] == user_id]['movie_id'].values

      # Lọc movies mà user chưa tương tác
      movies = movies[~movies['movie_id'].isin(user_movies)]

      # Xóa duplicates
      movies = movies.drop_duplicates(subset=['movie_id'])

      return torch.IntTensor(movies['movie_id'].values), torch.FloatTensor(movies.drop(columns=['movie_id']).values)
    
    @staticmethod
    def order(y_pred: np.ndarray, movies: pd.DataFrame, mode: Literal['explicit', 'implicit'], top_k=10) -> list[dict]:
      col_name= 'predicted_rating' if mode == 'explicit' else 'predicted_score'
      sorted_index = np.argsort(-y_pred, axis=0).reshape(-1).tolist()
      y_pred = y_pred[sorted_index]
      sorted_movies = movies.iloc[sorted_index]
      sorted_movies = sorted_movies.copy()
      sorted_movies[col_name] = y_pred if mode == 'implicit' else y_pred * 5
      sorted_movies.reset_index(drop=True, inplace=True)
      sorted_movies[col_name] = sorted_movies[col_name].apply(lambda x: round(x, 2))

      return sorted_movies.head(top_k)
    
    @staticmethod
    def plot_metrics(history: dict, title: str, figsize: tuple=(12, 4)) -> None:
      fig, ax = plt.subplots(1, 2, figsize=figsize)
      ax[0].plot(history['loss'], label='Train Loss')
      ax[0].plot(history['val_loss'], label='Validation Loss')
      ax[0].set_title('Training and Validation Loss')
      ax[0].set_xlabel('Epoch')
      ax[0].set_ylabel('Loss')
      ax[0].legend()

      # Plot metrics
      for metric, values in history.items():
        if metric not in ['loss', 'val_loss']:
          ax[1].plot(values, label=metric)
          
      ax[1].set_title('Metrics')
      ax[1].set_xlabel('Epoch')
      ax[1].set_ylabel('Value')
      ax[1].legend()
      plt.suptitle(title)
      plt.show()

class EarlyStopping:
  def __init__(self, patience=3, delta=0, verbose=False, path='checkpoint.pth') -> None:
    self.patience = patience
    self.delta = delta
    self.verbose = verbose
    self.path = path
    self.best_score = None
    self.early_stop = False
    self.counter = 0
    self.best_loss = float('inf')

  def __call__(self, val_loss, model: nn.Module) -> None:
    score = -val_loss

    if not self.best_score:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
      self.counter += 1
      print(f'Đếm ngược: {self.counter} out of {self.patience}') if self.verbose else None
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
      self.counter = 0

  def save_checkpoint(self, val_loss, model: nn.Module) -> None:
    print(f'Validation loss giảm ({self.best_loss:.6f} --> {val_loss:.6f}).  Lưu model ...') if self.verbose else None
    torch.save(model.state_dict(), self.path)
    self.best_loss = val_loss

age = [
  1, 18, 25, 35, 45, 50, 56
]

occupation = [
  'other', 'educator', 'artist', 'clerical', 'grad student',
  'customer service', 'doctor', 'executive', 'farmer', 'homemaker',
  'K-12 student', 'lawyer', 'programmer', 'retired', 'sales', 'scientist',
  'self-employed', 'engineer', 'craftsman', 'unemployed', 'writer'
]

genre = [
  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

cols_dict = {
  'ratings': ['user_id', 'movie_id', 'rating', 'timestamp'],
  'users': ['user_id', 'gender', 'age', 'occupation', 'zip_code'],
  'items': ['movie_id', 'title', 'genre'],
}

css = """
  <style>
    .card-container {
      display: flex;
      flex-direction: row;
      justify-content: center;
      align-items: start;
      gap: 20px;
      flex-wrap: wrap;
      margin: 20px 0;
    }

    .card {
      width: 100%;
      max-width: 300px;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      background-color: #eee;
      transition: transform 0.2s ease-in-out;
    }

    .card:hover {
      transform: scale(1.05);
    }

    .card-title {
      font-size: 1.25em;
      margin-bottom: 8px;
      color: #333;
    }

    .card-text {
      font-size: 1em;
      margin-bottom: 8px;
      color: #555;
    }

    .footer {
      position: fixed;
      left: 0;
      bottom: 0;
      width: 100%;
      background-color: rgb(45, 38, 48);
      color: #fff;
      text-align: center;
      padding: 10px;
    }
  </style>
"""