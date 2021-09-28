import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import tensorflow as tf
import joblib

from feats import get_feat, get_time_stock_feat
from config import GBM_FEATS, STOCK_TO_IDX
from nn.net import OptiverNet
from nn.dataset import OptiverTestDataset
# -------------------------------------------------------------------------------------------------


DATA_DIR = "../input/optiver-realized-volatility-prediction"
TEST_FILE = f"{DATA_DIR}/test.csv"

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predicting with {device}")
# -------------------------------------------------------------------------------------------------

print("Loading data and feature engineering...")

test = pd.read_csv(TEST_FILE, dtype=dtypes)
test_ids = test.stock_id.unique()

df_test = get_feat(stock_ids=test_ids, is_train=False, n_jobs=-1)
test = test[["row_id", "stock_id", "time_id"]].merge(df_test, on="row_id", how="left")
test["stock_id_idx"] = test["stock_id"].map(STOCK_TO_IDX)
del df_test
# -------------------------------------------------------------------------------------------------
# Tree

MODEL_DIR = "../input/optiver-realized-helper/tree"

print("Prediction...")
print(f"Number of features: {len(GBM_FEATS)}")

x_test = get_time_stock_feat(test)[GBM_FEATS]
stack_test = np.zeros(x_test.shape[0])

for idx in range(n_splits):
    start = time.perf_counter()
    model_path = os.path.join(MODEL_DIR, f"{idx + 1}.txt")
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

model1_pred = stack_test.copy()
# -------------------------------------------------------------------------------------------------
# Preparation for NN

test.fillna(0, inplace=True)
test = get_time_stock_feat(test)

test["stock_id_idx"] = test["stock_id"].map(STOCK_TO_IDX)
test.fillna(0, inplace=True)

cat_feats = ["stock_id_idx"]
num_feats = test.drop(["row_id", "stock_id", "time_id", "stock_id_idx"], axis=1).columns.tolist()
feats = cat_feats + num_feats
# -------------------------------------------------------------------------------------------------
# Torch with stock embedding

MODEL_DIR = "../input/optiver-realized-helper/torch"

qt_map = joblib.load(os.path.join(MODEL_DIR, "qt_map.pkl"))
scaler_map = joblib.load(os.path.join(MODEL_DIR, "scaler_map.pkl"))

x_test = test.copy()

for feat in num_feats:
    x_test[feat] = qt_map[feat].transform(x_test[[feat]])

for feat in num_feats:
    x_test[feat] = scaler_map[feat].transform(x_test[[feat]])

cat_feats_tensor = torch.LongTensor(x_test[cat_feats].values)
num_feats_tensor = torch.FloatTensor(x_test[num_feats].values)


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

model2_pred = stack_test.copy()
# -------------------------------------------------------------------------------------------------
# Tensorflow with auto encoder

MODEL_DIR = "../input/optiver-realized-helper/keras"

qt_map = joblib.load(os.path.join(MODEL_DIR, "qt_map.pkl"))
scaler_map = joblib.load(os.path.join(MODEL_DIR, "scaler_map.pkl"))

x_test = test.copy()

for feat in num_feats:
    x_test[feat] = qt_map[feat].transform(x_test[[feat]])

for feat in num_feats:
    x_test[feat] = scaler_map[feat].transform(x_test[[feat]])


print("Prediction...")
print(f"Number of features: {len(feats)}")

test_cats = x_test.loc[:, cat_feats]
test_nums = x_test.loc[:, num_feats]

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

model3_pred = stack_test.copy()
# -------------------------------------------------------------------------------------------------
# Output submission file

test["target"] = 0.2 * model1_pred + 0.4 * model2_pred + 0.4 * model3_pred
test[["row_id", "target"]].to_csv("submission.csv", index=False)
