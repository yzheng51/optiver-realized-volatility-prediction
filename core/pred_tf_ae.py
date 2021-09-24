import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from feats import get_feat, get_time_stock_feat
from config import STOCK_TO_IDX
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TEST_FILE = f"{DATA_DIR}/test.csv"
MODEL_DIR = "../input/orvpmodelkeras"

dtypes = {"stock_id": "int16", "time_id": "int32", "target": "float64"}

n_splits = 5


def build_model():
    embed_size = 16
    hidden_units = (128, 64, 32, 16)

    cat_input = tf.keras.Input(shape=(len(cat_feats), ), name="cat_feats")
    num_input = tf.keras.Input(shape=(len(num_feats), ), name="num_feats")

    encoded = tf.keras.layers.BatchNormalization()(cat_input)
    encoded = tf.keras.layers.GaussianNoise(0.03)(encoded)
    encoded = tf.keras.layers.Dense(embed_size, activation="swish")(encoded)

    decoded = tf.keras.layers.Dropout(0.03)(encoded)
    decoded = tf.keras.layers.Dense(len(STOCK_TO_IDX), activation="softmax", name="decode")(decoded)

    x = tf.keras.layers.Concatenate()([encoded, num_input])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.05)(x)

    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = tf.keras.models.Model(inputs=[cat_input, num_input], outputs=[decoded, out])

    return model
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
# -------------------------------------------------------------------------------------------------


print("Prediction...")
print(f"Number of features: {len(feats)}")

test_cats = test.loc[:, cat_feats]
test_nums = test.loc[:, num_feats]

model = build_model()
stack_test = np.zeros(test.shape[0])

for idx in range(n_splits):
    start = time.perf_counter()

    model_path = os.path.join(MODEL_DIR, f"{idx + 1}.h5")
    model.load_weights(model_path)
    pred = model.predict([test_cats, test_nums], batch_size=2048)[-1].ravel()

    stack_test += pred
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
