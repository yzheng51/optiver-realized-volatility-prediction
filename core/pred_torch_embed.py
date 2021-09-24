import os
import time

import numpy as np
import pandas as pd
import torch
import joblib

from feats import get_feat, get_time_stock_feat
from config import STOCK_TO_IDX
from nn.net import OptiverNet
from nn.dataset import OptiverTestDataset
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TEST_FILE = f"{DATA_DIR}/test.csv"
MODEL_DIR = "../input/orvpmodelpytorch"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predicting with {device}")
# -------------------------------------------------------------------------------------------------


print("Loading data and feature engineering...")

test = pd.read_csv(TEST_FILE, dtype=dtypes)
test_ids = test.stock_id.unique()

df_test = get_feat(stock_ids=test_ids, is_train=False, n_jobs=-1)
test = test[["row_id", "stock_id", "time_id"]].merge(df_test, on="row_id", how="left")
del df_test

test.fillna(0, inplace=True)
test = get_time_stock_feat(test)

test["stock_id_idx"] = test["stock_id"].map(STOCK_TO_IDX)
test.fillna(0, inplace=True)

cat_feats = ["stock_id_idx"]
num_feats = test.drop(["row_id", "stock_id", "time_id", "stock_id_idx"], axis=1).columns.tolist()
feats = cat_feats + num_feats

qt_map = joblib.load(os.path.join(MODEL_DIR, "qt_map.pkl"))
scaler_map = joblib.load(os.path.join(MODEL_DIR, "scaler_map.pkl"))

for feat in num_feats:
    test[feat] = qt_map[feat].transform(test[[feat]])

for feat in num_feats:
    test[feat] = scaler_map[feat].transform(test[[feat]])

cat_feats_tensor = torch.LongTensor(test[cat_feats].values)
num_feats_tensor = torch.FloatTensor(test[num_feats].values)
# -------------------------------------------------------------------------------------------------


print("Prediction...")
print(f"Number of features: {len(feats)}")

test_data = OptiverTestDataset(cat_feats_tensor, num_feats_tensor)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, drop_last=False, batch_size=2048)

model = OptiverNet(vocab_size=len(STOCK_TO_IDX), embed_size=16, feat_size=len(num_feats))
model = model.to(device)

stack_test = np.zeros(test.shape[0])

for idx in range(n_splits):
    start = time.perf_counter()

    model_path = os.path.join(MODEL_DIR, f"{idx + 1}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_pred_fold = list()
    model.eval()
    with torch.no_grad():
        for cats, nums in test_loader:
            cats, nums = cats.to(device), nums.to(device)
            output = model(cats, nums)
            y_pred_fold.append(output.cpu().numpy().ravel())
    stack_test += np.hstack(y_pred_fold)
    duration = time.perf_counter() - start
    print(
        f"Fold: {idx + 1}/{n_splits}... ",
        f"Model path: `{model_path}`...",
        f"Elapse time: {duration:.2f}s...",
    )
stack_test /= n_splits

test["target"] = stack_test
test[["row_id", "target"]].to_csv("submission.csv", index=False)
test[["row_id", "target"]]
