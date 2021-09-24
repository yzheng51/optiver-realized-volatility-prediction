import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from feats import get_feat, get_time_stock_feat
from config import GBM_FEATS
from utils.timer import Timer
from utils.evaluate import rmspe, lgb_rmspe, lgb_rmse
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
TEST_FILE = f"{DATA_DIR}/test.csv"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5

timer = Timer()
# -------------------------------------------------------------------------------------------------


print("Loading data and feature engineering...")
timer.start()

train = pd.read_csv(TRAIN_FILE, dtype=dtypes)
train["row_id"] = train["stock_id"].astype("str") + "-" + train["time_id"].astype("str")
train_ids = train.stock_id.unique()

df_train = get_feat(stock_ids=train_ids, is_train=True, n_jobs=-1)
train = train[["row_id", "stock_id", "time_id", "target"]].merge(df_train, on="row_id", how="left")
del df_train

train = get_time_stock_feat(train)

timer.stop()
# -------------------------------------------------------------------------------------------------


print("Training model...")
timer.start()

print(f"Number of features: {len(GBM_FEATS)}")
x_train = train[GBM_FEATS]
y_train = train["target"].values

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1024)

stack_train = np.zeros(x_train.shape[0])
feats_impt = np.zeros(len(GBM_FEATS))
rmspe_mean = 0


for i, (train_idx, valid_idx) in enumerate(kfold.split(x_train)):
    print(f"Fold: {i + 1}/{n_splits}... ")
    model = lgb.LGBMRegressor(
        boosting="gbdt",
        importance_type="gain",
        learning_rate=0.01,
        max_depth=7,
        num_leaves=80,
        min_child_weight=20,
        n_estimators=3000,
        lambda_l2=1,
        bagging_freq=1,
        bagging_fraction=0.6,
        feature_fraction=0.3,
        random_state=55384,
        bagging_seed=13881,
        n_jobs=50,
    )
    model.fit(
        x_train.iloc[train_idx], y_train[train_idx],
        sample_weight=1 / np.square(y_train[train_idx]),
    )
    y_valid = model.predict(x_train.iloc[valid_idx])

    stack_train[valid_idx] = y_valid
    feats_impt += model.feature_importances_ / n_splits
    rmspe_mean += rmspe(y_train[valid_idx], y_valid)
    model.booster_.save_model(f"../models/{i + 1}.txt")
    print(f"RMSPE {rmspe(y_train[valid_idx], y_valid):.6f}")

feats_impt /= n_splits
rmspe_mean /= n_splits

np.save("../models/stack_train.npy", stack_train)

importance = pd.DataFrame(zip(GBM_FEATS, feats_impt), columns=["feat", "score"]) \
    .sort_values("score", ascending=False) \
    .reset_index(drop=True)
importance.to_csv("../models/impt.csv", index=False)

timer.stop()

print(f"RMSPE mean is: {rmspe_mean:.6f}")
