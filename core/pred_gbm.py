import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from feats import get_feat, get_time_stock_feat
from config import GBM_FEATS, STOCK_TO_IDX
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TEST_FILE = f"{DATA_DIR}/test.csv"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5
# -------------------------------------------------------------------------------------------------


print("Loading data and feature engineering...")

test = pd.read_csv(TEST_FILE, dtype=dtypes)
test_ids = test.stock_id.unique()

df_test = get_feat(stock_ids=test_ids, is_train=False, n_jobs=-1)
test = test[["row_id", "stock_id", "time_id"]].merge(df_test, on="row_id", how="left")
del df_test

test = get_time_stock_feat(test)
test["stock_id_idx"] = test["stock_id"].map(STOCK_TO_IDX)
# -------------------------------------------------------------------------------------------------


print("Prediction...")
print(f"Number of features: {len(GBM_FEATS)}")

x_test = test[GBM_FEATS]
stack_test = np.zeros(x_test.shape[0])

for idx in range(n_splits):
    start = time.perf_counter()
    model_path = f"../input/orvplgb/{idx + 1}.txt"
    model = lgb.Booster(model_file=model_path)
    y_pred = model.predict(x_test)

    stack_test += y_pred
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
