import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
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

feats = [
    "book_time_id_count",
    "book_wap1_mean",
    "book_wap1_std",
    "book_wap1_max_sub_min",
    "book_wap2_mean",
    "book_wap2_std",
    "book_wap2_max_sub_min",
    "book_log_return_wap1_mean",
    "book_log_return_wap1_std",
    "book_log_return_wap1_abs_sum",
    "book_log_return_wap1_rv",
    "book_log_return_wap2_mean",
    "book_log_return_wap2_std",
    "book_log_return_wap2_abs_sum",
    "book_log_return_wap2_rv",
    "book_wap_balance_mean",
    "book_wap_balance_std",
    "book_wap_balance_abs_sum",
    "book_price_spread1_sum",
    "book_price_spread1_mean",
    "book_price_spread1_std",
    "book_price_spread2_sum",
    "book_price_spread2_mean",
    "book_price_spread2_std",
    "book_bid_spread_sum",
    "book_bid_spread_mean",
    "book_bid_spread_std",
    "book_ask_spread_sum",
    "book_ask_spread_mean",
    "book_ask_spread_std",
    "book_bid_ask_spread_mean",
    "book_bid_ask_spread_std",
    "book_bid_ask_spread_abs_sum",
    "book_total_volume_sum",
    "book_total_volume_mean",
    "book_total_volume_std",
    "book_volume_imbalance_mean",
    "book_volume_imbalance_std",
    "book_volume_imbalance_abs_sum",
    "book_log_return_wap1_rv_diff",
    "book_log_return_wap1_rv_pct_change",
    "book_log_return_wap2_rv_diff",
    "book_log_return_wap2_rv_pct_change",
    "book_last_300_log_return_wap1_rv",
    "book_last_300_log_return_wap2_rv",
    "trade_time_id_count",
    "trade_price_mean",
    "trade_price_std",
    "trade_price_max_sub_min",
    "trade_log_return_mean",
    "trade_log_return_std",
    "trade_log_return_abs_sum",
    "trade_log_return_rv",
    "trade_size_sum",
    "trade_size_max_sub_min",
    "trade_order_count_sum",
    "trade_order_count_max_sub_min",
    "trade_amount_sum",
    "trade_amount_mean",
    "trade_amount_std",
    "trade_amount_max_sub_min",
    "trade_size_div_order_count_mean",
    "trade_size_div_order_count_std",
    "trade_seconds_diff_mean",
    "trade_seconds_diff_std",
    "trade_seconds_diff_abs_sum",
    "trade_seconds_diff_rv",
    "trade_log_return_rv_diff",
    "trade_log_return_rv_pct_change",
    "trade_last_300_log_return_rv",
    "trade_last_300_seconds_diff_rv",
    "time_id_book_log_return_wap1_rv_mean",
    "time_id_book_log_return_wap1_rv_std",
    "time_id_book_log_return_wap2_rv_mean",
    "time_id_book_log_return_wap2_rv_std",
    "time_id_trade_log_return_rv_mean",
    "time_id_trade_log_return_rv_std",
    "time_id_trade_seconds_diff_rv_mean",
    "time_id_trade_seconds_diff_rv_std",
    "time_id_book_last_300_log_return_wap1_rv_mean",
    "time_id_book_last_300_log_return_wap1_rv_std",
    "time_id_book_last_300_log_return_wap2_rv_mean",
    "time_id_book_last_300_log_return_wap2_rv_std",
    "time_id_trade_last_300_log_return_rv_mean",
    "time_id_trade_last_300_log_return_rv_std",
    "time_id_trade_last_300_seconds_diff_rv_mean",
    "time_id_trade_last_300_seconds_diff_rv_std",
    "stock_id_book_log_return_wap1_rv_mean",
    "stock_id_book_log_return_wap1_rv_std",
    "stock_id_book_log_return_wap2_rv_mean",
    "stock_id_book_log_return_wap2_rv_std",
    "stock_id_trade_log_return_rv_mean",
    "stock_id_trade_log_return_rv_std",
    "stock_id_trade_seconds_diff_rv_mean",
    "stock_id_trade_seconds_diff_rv_std",
    "stock_id_book_last_300_log_return_wap1_rv_mean",
    "stock_id_book_last_300_log_return_wap1_rv_std",
    "stock_id_book_last_300_log_return_wap2_rv_mean",
    "stock_id_book_last_300_log_return_wap2_rv_std",
    "stock_id_trade_last_300_log_return_rv_mean",
    "stock_id_trade_last_300_log_return_rv_std",
    "stock_id_trade_last_300_seconds_diff_rv_mean",
    "stock_id_trade_last_300_seconds_diff_rv_std",
]
print(f"Number of features: {len(feats)}")
x_train = train[feats]
y_train = train["target"].values

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1024)

stack_train = np.zeros(x_train.shape[0])
feats_impt = np.zeros(len(feats))
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

importance = pd.DataFrame(zip(feats, feats_impt), columns=["feat", "score"]) \
    .sort_values("score", ascending=False) \
    .reset_index(drop=True)
importance.to_csv("../models/impt.csv", index=False)

timer.stop()

print(f"RMSPE mean is: {rmspe_mean:.6f}")
