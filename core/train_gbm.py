import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed

from feats import get_book_feat, get_trade_feat, get_time_stock_feat
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


def get_feat(stock_ids, is_train=True, n_jobs=-1):
    def ufunc(stock_id):
        if is_train:
            file_path_book = f"{DATA_DIR}/book_train.parquet/stock_id={stock_id}"
            file_path_trade = f"{DATA_DIR}/trade_train.parquet/stock_id={stock_id}"
        else:
            file_path_book = f"{DATA_DIR}/book_test.parquet/stock_id={stock_id}"
            file_path_trade = f"{DATA_DIR}/trade_test.parquet/stock_id={stock_id}"

        return pd.merge(get_book_feat(file_path_book), get_trade_feat(file_path_trade), on="row_id", how="left")

    df = Parallel(n_jobs=n_jobs, verbose=1)(delayed(ufunc)(stock_id) for stock_id in stock_ids)
    df = pd.concat(df, ignore_index=True)

    return df
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


print("Traing model...")
timer.start()

feats = train.columns.drop(["row_id", "stock_id", "time_id", "target"]).tolist()
x_train = train[feats]
y_train = train["target"].values

kfold = GroupKFold(n_splits=n_splits)

stack_train = np.zeros(x_train.shape[0])
feats_impt = np.zeros(len(feats))
rmspe_mean = 0


for i, (train_idx, valid_idx) in enumerate(kfold.split(x_train, y_train, train["time_id"])):
    print(f'Stacking: {i + 1}/{n_splits}... ')
    model = lgb.LGBMRegressor(
        boosting="gbdt",
        importance_type='gain',
        learning_rate=0.01,
        max_depth=8,
        num_leaves=60,
        min_child_weight=20,
        n_estimators=2000,
        lambda_l2=1,
        bagging_freq=1,
        bagging_fraction=0.7,
        feature_fraction=0.5,
        random_state=54786,
        bagging_seed=12783,
        n_jobs=50,
    )
    model.fit(
        x_train.iloc[train_idx], y_train[train_idx],
        sample_weight=1 / np.square(y_train[train_idx]),
        eval_set=[(x_train.iloc[valid_idx], y_train[valid_idx])],
        eval_metric=lambda y_true, y_pred: [lgb_rmspe(y_true, y_pred), lgb_rmse(y_true, y_pred)],
        verbose=200,
    )
    y_valid = model.predict(x_train.iloc[valid_idx])

    stack_train[valid_idx] = y_valid
    feats_impt += model.feature_importances_ / n_splits
    rmspe_mean += rmspe(y_train[valid_idx], y_valid)
    model.booster_.save_model(f"../models/{i + 1}.txt")

feats_impt /= n_splits
rmspe_mean /= n_splits

np.save("../models/stack_train.npy", stack_train)

importance = pd.DataFrame(zip(feats, feats_impt), columns=["feat", "score"]) \
    .sort_values("score", ascending=False) \
    .reset_index(drop=True)
importance.to_csv("../models/impt.csv", index=False)

print(f"RMSPE mean is: {rmspe_mean:.6f}")
