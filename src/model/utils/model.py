import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, roc_auc_score

import numpy as np
from math import sqrt
from .utils import Utils, EarlyStopping
from typing import Literal

__model_version__ = '1.0.2'

from typing import Literal

class NCF(nn.Module):
  
  def __init__(self, mode: Literal['explicit', 'implicit'], num_users=6040, num_items=3952, user_dim=48, item_dim=19, num_factors=32, criterion=None, dropout=0.1, lr=1e-3, weight_decay=1e-5, verbose=False, gpu=True):
    super(NCF, self).__init__()
    self.mode = mode
    self.num_users = num_users
    self.num_items = num_items
    
    """
    Đây là embedding layers cho MLP(MLP Layers) và MF(Matrix Factorization Layers)
    +1 vì embedding layer bắt đầu từ index 0
    num_factors là số lượng embedding vectors
    """
    self.user_embedding_mlp = nn.Embedding(num_users + 1, num_factors)
    self.item_embedding_mlp = nn.Embedding(num_items + 1, num_factors)

    self.user_embedding_mf = nn.Embedding(num_users + 1, num_factors)
    self.item_embedding_mf = nn.Embedding(num_items + 1, num_factors)

    """
    Đây là embedding layers cho user features
    Mục đích là để tăng độ phức tạp của mô hình bằng cách thêm các features của user
    """
    self.user_features = nn.Sequential(
      nn.Linear(user_dim, num_factors*2),
      nn.ReLU(),
      nn.Linear(num_factors*2, num_factors),
      nn.ReLU()
    )

    self.item_features = nn.Sequential(
      nn.Linear(item_dim, num_factors*2),
      nn.ReLU(),
      nn.Linear(num_factors*2, num_factors),
      nn.ReLU()
    )
  
    """
    Đây là MLP layers
    """
    self.MLP = nn.Sequential(
      nn.Linear(num_factors * 4, 512),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_factors),
      nn.ReLU()
    )

    self.neu_mf = nn.Linear(num_factors * 2, 1)

    self.sigmoid = nn.Sigmoid()
    self.criterion = criterion
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    self.__init_weights()
    self.to(self.device)

    print(self, '\nĐang chạy trên: ', self.device) if verbose else None
    print(f'Số lượng parameters: {self.params_count():,}') if verbose else None


  """
  Đây là hàm để khởi tạo dữ liệu và chuyển sang tensor
  input là dữ liệu đầu vào
  y là dữ liệu đầu ra
  """
  def __init_data(self, input, y=None):
    # Khởi tạo dữ liệu
    X_user_id, X_item_id, X_user, X_item = input

    # Chuyển sang tensor
    X_user =torch.FloatTensor(X_user).to(self.device, non_blocking=True)
    X_item = torch.FloatTensor(X_item).to(self.device, non_blocking=True)
    # Chuyển sang tensor long
    X_user_id = torch.LongTensor(X_user_id).to(self.device, non_blocking=True)
    X_item_id = torch.LongTensor(X_item_id).to(self.device, non_blocking=True)

    # Chuyển sang tensor float
    y = torch.FloatTensor(y).to(self.device, non_blocking=True) if y is not None else None

    return X_user, X_item, X_user_id, X_item_id, y

  def __init_weights(self) -> None:
    # Khởi tạo weights
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01) # Normat initialization với mean=0 và standard deviation=0.01

    for m in self.MLP.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=1, mode='fan_in', nonlinearity='relu') # Quá trình khởi tạo fan_in, bảo toàn tính tương thích cho ReLU
        m.bias.data.fill_(0.01)
    
    nn.init.xavier_uniform_(self.neu_mf.weight, gain=nn.init.calculate_gain('sigmoid'))

  def forward(self, user_id: torch.IntTensor, item_id: torch.IntTensor, user_features: torch.FloatTensor, item_features: torch.FloatTensor, weights=None) -> torch.Tensor:
    if not weights:
      user_embedding_mlp = self.user_embedding_mlp(user_id) # (batch_size, num_factors)
      user_embedding_mf = self.user_embedding_mf(user_id) # (batch_size, num_factors)
    else:
      user_embedding_mlp = torch.tensor(weights[0], device=self.device).repeat(len(user_id), 1)
      user_embedding_mf = torch.tensor(weights[1], device=self.device).repeat(len(user_id), 1)

    item_embedding_mlp = self.item_embedding_mlp(item_id) # (batch_size, num_factors)
    item_embedding_mf = self.item_embedding_mf(item_id) # (batch_size, num_factors)

    user_features = self.user_features(user_features) # (batch_size, num_factors)
    item_features = self.item_features(item_features) # (batch_size, num_factors)

    user_embedding_mlp = torch.cat([user_embedding_mlp, user_features], dim=-1) # (batch_size, num_factors * 2)
    item_embedding_mlp = torch.cat([item_embedding_mlp, item_features], dim=-1) # (batch_size, num_factors * 2)

    mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1) # (batch_size, num_factors * 4)
    mlp_output = self.MLP(mlp_input) # (batch_size, num_factors)

    mf_output = torch.mul(user_embedding_mf, item_embedding_mf) # (batch_size, num_factors)

    neu_mf_input = torch.cat([mlp_output, mf_output], dim=-1) # (batch_size, num_factors * 2)
    neu_mf = self.neu_mf(neu_mf_input) # (batch_size, 1)

    return self.sigmoid(neu_mf).flatten()

  
  def fit(self, X: list[np.ndarray], y: np.ndarray, epochs: int, batch_size: int, X_val: list[np.ndarray] = None, y_val: np.ndarray = None, k:int = None, scheduler: torch.optim.lr_scheduler = None, early_stopping: EarlyStopping = None) -> dict:
    X_user_id, X_item_id, X_user, X_item, y = self.__init_data(X, y)
    dataset = TensorDataset(X_user_id, X_item_id, X_user, X_item, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    self.train()

    losses = []
    all_metrics = []
    lrs = []
    for epoch in range(epochs):
      total_loss = 0.0
      for i, batch in enumerate(dataloader):
        X_user_id, X_item_id, X_user, X_item, y = batch

        self.optimizer.zero_grad() # Zero the gradients
        output = self(X_user_id, X_item_id, X_user, X_item) # Forward pass
        loss = self.criterion(output, y) # Compute the loss
        loss.backward() # Backward pass
        self.optimizer.step() # Update the weights

        total_loss += loss.item()

        print(f'Lần huấn luyện thứ {epoch+1}/{epochs}') if i == 0 else None
        print(f'Batch {i+1}/{len(dataloader)} - loss: {(total_loss / (i+1)):.4f}', end='\r' if i + 1 < len(dataloader) else ' ')
      
      losses.append(total_loss / (i + 1))
      # Kiểm tra nếu có dữ liệu validation
      if X_val is not None and y_val is not None:
        self.eval()
        metrics = self.evaluate(X_val, y_val, batch_size, k)

        if scheduler:
          scheduler.step(metrics[0])
          lrs.append(scheduler.get_last_lr()[0])

        if self.mode == 'explicit':
          all_metrics.append([*metrics])
          print(f'- Val Loss: {metrics[0]:.4f} - R2: {metrics[1]:.4f} - MAE: {metrics[2]:.4f} - MSE: {metrics[3]:.4f} - RMSE: {metrics[4]:.4f} - lr: {lrs[-1]}')
        else:
          all_metrics.append([*metrics])
          print(f'- Val Loss: {metrics[0]:.4f} - NDCG: {metrics[1]:.4f} - HR: {metrics[2]:.4f} - ROC-AUC: {metrics[3]:.4f} - lr: {lrs[-1]}')

        if early_stopping:
          early_stopping(metrics[0], self)
          if early_stopping.early_stop:
            print("Dừng huấn luyện")
            break

        self.train()
      else:
        print()
    if self.mode == 'explicit':
      history = {
        'loss': losses,
        'val_loss': [m[0] for m in all_metrics],
        'r2': [m[1] for m in all_metrics],
        'mae': [m[2] for m in all_metrics],
        'mse': [m[3] for m in all_metrics],
        'rmse': [m[4] for m in all_metrics],
        'lr': lrs
      }
    else:
      history = {
        'loss': losses,
        'val_loss': [m[0] for m in all_metrics],
        'ndcg': [m[1] for m in all_metrics],
        'hr': [m[2] for m in all_metrics],
        'roc_auc': [m[3] for m in all_metrics],
        'lr': lrs
      }

    return history

  def predict(self, user_id: torch.IntTensor, item_id:torch.IntTensor, user: torch.FloatTensor, items: torch.FloatTensor)-> torch.Tensor:   
      user_id, item_id, user_input, item_input, _ = self.__init_data([user_id, item_id, user, items])
      return self(user_id, item_id, user_input, item_input)

  def evaluate(self, X_val: list, y_val: np.ndarray, batch_size: int, k: int = None) -> tuple[float]:

    user_id, item_id, user_input, item_input, target = self.__init_data(X_val, y_val)

    dataset = TensorDataset(user_id, item_id, user_input, item_input, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    avg_loss, y_preds = 0.0, []
    with torch.no_grad():
      for i, batch in enumerate(dataloader):
        user_id, item_id, user_input, item_input, y = batch

        preds = self(user_id, item_id, user_input, item_input)

        y_preds.extend(preds.cpu().numpy())

        loss = self.criterion(preds, y)
        avg_loss += loss.item()

    avg_loss /= (i + 1)

    target = target.cpu().numpy()
    y_preds = np.array(y_preds).reshape(-1)

    if self.mode == 'explicit':
      r2 = r2_score(target, y_preds)
      mae = mean_absolute_error(target, y_preds)
      mse = mean_squared_error(target, y_preds)
      rmse = sqrt(mse)

      return avg_loss, r2, mae, mse, rmse
    else:
      ndcg, hr = Utils.ndcg_hit_ratio(y_preds, X_val[2], y_val, k)
      roc_auc = roc_auc_score(target, y_preds)

      return avg_loss, ndcg, hr, roc_auc

  def params_count(self)-> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  def save_weights(self, path)-> None:
    torch.save(self.state_dict(), path)

  def load_weights(self, path, eval=True)-> None:
    self.load_state_dict(torch.load(path, map_location=self.device))
    self.eval() if eval else None 
