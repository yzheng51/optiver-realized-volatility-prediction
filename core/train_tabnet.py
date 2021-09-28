import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.model_selection import KFold
import joblib

from feats import get_feat, get_time_stock_feat
from config import STOCK_TO_IDX
from utils.timer import Timer
from utils.evaluate import rmspe
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
TEST_FILE = f"{DATA_DIR}/test.csv"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5

def rmspe_loss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true)**2)).clone()


class RMSPEMetric(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        return rmspe(y_true, y_pred)


timer = Timer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training with {device}")
# -------------------------------------------------------------------------------------------------


print("Loading data and feature engineering...")
timer.start()

train = pd.read_csv(TRAIN_FILE, dtype=dtypes)
train["row_id"] = train["stock_id"].astype("str") + "-" + train["time_id"].astype("str")
train_ids = train.stock_id.unique()

df_train = get_feat(stock_ids=train_ids, is_train=True, n_jobs=-1)
train = train[["row_id", "stock_id", "time_id", "target"]].merge(df_train, on="row_id", how="left")
del df_train

train.fillna(0, inplace=True)
train = get_time_stock_feat(train)

train["stock_id_idx"] = train["stock_id"].map(STOCK_TO_IDX)

cat_feats = ["stock_id_idx"]
num_feats = train.drop(["row_id", "stock_id", "time_id", "target", "stock_id_idx"], axis=1).columns.tolist()
feats = cat_feats + num_feats

qt_map = dict()
scaler_map = dict()

for feat in num_feats:
    qt = QuantileTransformer(random_state=12, n_quantiles=2000, output_distribution="normal")
    train[feat] = qt.fit_transform(train[[feat]])
    qt_map[feat] = qt

for feat in num_feats:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train[feat] = scaler.fit_transform(train[[feat]])
    scaler_map[feat] = scaler

x_train = train[feats].values
y_train = train[["target"]].values

timer.stop()
# -------------------------------------------------------------------------------------------------


print("Training model...")
timer.start()

print(f"Number of features: {len(feats)}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=8626)

score_rmspe = [0] * n_splits
stack_train = np.zeros(x_train.shape[0])

for idx, (train_idx, valid_idx) in enumerate(kfold.split(x_train)):
    print(f"Fold: {idx + 1}/{n_splits}... ")
    model = TabNetRegressor(
        cat_idxs=[0],
        cat_dims=[len(STOCK_TO_IDX)],
        cat_emb_dim=1,
        n_d=17,
        n_a=16,
        n_steps=2,
        gamma=2,
        n_independent=2,
        n_shared=2,
        lambda_sparse=0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=(2e-2)),
        mask_type="entmax",
        scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        seed=42,
        verbose=10,
    )
    model.fit(
        x_train[train_idx], y_train[train_idx],
        eval_set=[(x_train[valid_idx], y_train[valid_idx])],
        max_epochs=200,
        patience=50,
        batch_size=1024 * 20,
        virtual_batch_size=128 * 20,
        num_workers=1,
        drop_last=True,
        eval_metric=[RMSPEMetric],
        loss_fn=rmspe_loss,
    )

    y_valid = model.predict(x_train[valid_idx]).ravel()

    stack_train[valid_idx] = y_valid
    score_rmspe[idx] = rmspe(y_train[valid_idx].ravel(), y_valid)
    saved_filepath = model.save_model(f"../models/{idx + 1}")

print(f"RMSE mean is: {np.mean(score_rmspe):.6f}")

timer.stop()

joblib.dump(qt_map, "../models/qt_map.pkl")
joblib.dump(scaler_map, "../models/scaler_map.pkl")
np.save("../models/stack_train.npy", stack_train)
