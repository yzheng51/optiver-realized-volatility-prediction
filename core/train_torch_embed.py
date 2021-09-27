import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.model_selection import KFold
import joblib

from feats import get_feat, get_time_stock_feat
from config import STOCK_TO_IDX
from nn.net import OptiverNet
from nn.dataset import OptiverTrainDataset
from nn.utils import RMSPELoss
from utils.timer import Timer
from utils.evaluate import rmspe
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
TEST_FILE = f"{DATA_DIR}/test.csv"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5

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

cat_feats_tensor = torch.LongTensor(train[cat_feats].values)
num_feats_tensor = torch.FloatTensor(train[num_feats].values)
y_train = torch.FloatTensor(train[["target"]].values)

timer.stop()
# -------------------------------------------------------------------------------------------------


print("Training model...")
timer.start()

print(f"Number of features: {len(feats)}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=5197)

epochs = 100
criterion = RMSPELoss()
score_rmspe = [0] * n_splits
stack_train = np.zeros(train.shape[0])

for idx, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    print(f"Fold {idx + 1}/{n_splits}")
    net = OptiverNet(vocab_size=len(STOCK_TO_IDX), embed_size=16, feat_size=len(num_feats))
    net.to(device)

    # data preparation
    train_data = OptiverTrainDataset(cat_feats_tensor[train_idx], num_feats_tensor[train_idx], y_train[train_idx])
    valid_data = OptiverTrainDataset(cat_feats_tensor[valid_idx], num_feats_tensor[valid_idx], y_train[valid_idx])
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, drop_last=True, batch_size=512)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, drop_last=False, batch_size=2048)

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.2, min_lr=1e-6)

    best_rmspe = 1e9
    early_stop_counter = 0
    print("|   Epoch  | Train Loss  | Valid RMSPE | Valid Loss  | Learning Rate | Epapsed Time |")
    print("| -------- | ----------- | ----------- | ----------- | ------------- | ------------ |")
    for e in range(epochs):
        start = time.perf_counter()
        train_losses = list()
        step_counter = 0
        net.train()
        for ii, (cats, nums, labels) in enumerate(train_loader):
            cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)

            output = net(cats, nums)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        # validation
        net.eval()
        valid_array = list()
        valid_losses = list()
        with torch.no_grad():
            for cats, nums, labels in valid_loader:
                cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)

                output = net(cats, nums)
                loss = criterion(output, labels)

                valid_losses.append(loss.item())
                valid_array.append(output.cpu().numpy().ravel())
        valid_array = np.concatenate(valid_array)

        # output train information
        score = rmspe(y_train[valid_idx].numpy().ravel(), valid_array)
        scheduler.step(score)
        duration = time.perf_counter() - start
        print(
            "|  {}/{} ".format(e + 1, epochs).ljust(10),
            "|   {:.4f}  ".format(np.mean(train_losses)).ljust(13),
            "|   {:.4f}  ".format(score).ljust(13),
            "|   {:.4f}  ".format(np.mean(valid_losses)).ljust(13),
            "|   {:.6f}  ".format(scheduler._last_lr[-1]).ljust(15),
            "|  {:.2f}s ".format(duration).ljust(14),
            "|"
        )
        if best_rmspe > score:
            best_rmspe = score
            model_path = os.path.join(f"../models/{idx + 1}.pth")
            torch.save(net.state_dict(), model_path)
            stack_train[valid_idx] = valid_array
            score_rmspe[idx] = best_rmspe
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= 10:
            break
    print(f"RMSPE is {best_rmspe:.6f}\n")

print(f"RMSE mean is: {np.mean(score_rmspe):.6f}")

timer.stop()

joblib.dump(qt_map, "../models/qt_map.pkl")
joblib.dump(scaler_map, "../models/scaler_map.pkl")
np.save("../models/stack_train.npy", stack_train)
