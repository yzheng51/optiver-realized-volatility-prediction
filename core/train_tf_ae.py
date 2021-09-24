import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, OneHotEncoder
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


def keras_rmspe(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_true - y_pred) / y_true)))


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

timer.stop()
# -------------------------------------------------------------------------------------------------


print("Training model...")
timer.start()

print(f"Number of features: {len(feats)}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=2021)

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_pred_loss", patience=10, verbose=0, mode="min", restore_best_weights=True
)
plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_pred_loss", factor=0.2, patience=4, verbose=0, mode="min", min_lr=1e-6
)

enc = OneHotEncoder()
enc.fit(train[cat_feats])

score_rmspe = [0] * n_splits
stack_train = np.zeros(train.shape[0])

for idx, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    print("Fold: {}/{}".format(idx + 1, n_splits))

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
        loss={
            "decode": tf.keras.losses.CategoricalCrossentropy(),
            "pred": keras_rmspe,
        },
    )

    train_cats = train.loc[train_idx, cat_feats]
    train_nums = train.loc[train_idx, num_feats]

    valid_cats = train.loc[valid_idx, cat_feats]
    valid_nums = train.loc[valid_idx, num_feats]

    y_train = train.loc[train_idx, "target"].values
    y_valid = train.loc[valid_idx, "target"].values

    checkpoint_filepath = f'../models/{idx + 1}.h5'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    model.fit(
        [train_cats, train_nums],
        [enc.transform(train_cats).A, y_train],
        batch_size=512,
        epochs=100,
        validation_data=([valid_cats, valid_nums], [enc.transform(valid_cats).A, y_valid]),
        callbacks=[es, plateau, model_checkpoint_callback],
        validation_batch_size=len(y_valid),
        shuffle=True,
        verbose=1,
    )
    model.load_weights(checkpoint_filepath)
    y_pred = model.predict([valid_cats, valid_nums], batch_size=2048)[-1].ravel()
    stack_train[valid_idx] = y_pred
    score = rmspe(y_true=y_valid, y_pred=y_pred)
    score_rmspe[idx] = score

    print(f"Fold {idx + 1}/{n_splits}: {score:.5f}\n")

print(f"RMSE mean is: {np.mean(score_rmspe):.6f}")

timer.stop()

joblib.dump(qt_map, "../models/qt_map.pkl")
joblib.dump(scaler_map, "../models/scaler_map.pkl")
np.save("../models/stack_train.npy", stack_train)
