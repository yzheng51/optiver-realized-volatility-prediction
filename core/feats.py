import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def calc_wap(bpr, bsz, apr, asz):
    return (bpr * asz + apr * bsz) / (bsz + asz)


def log_return(x):
    return np.log(x).diff()


def rv(x):
    return np.sqrt(np.sum(x**2))


def abs_sum(x):
    return np.sum(np.abs(x))


def max_sub_min(x):
    return np.max(x) - np.min(x)


def get_book_feat(file_path):
    df = pd.read_parquet(file_path)

    bpr1, bsz1, apr1, asz1 = (df[col].values for col in ["bid_price1", "bid_size1", "ask_price1", "ask_size1"])
    bpr2, bsz2, apr2, asz2 = (df[col].values for col in ["bid_price2", "bid_size2", "ask_price2", "ask_size2"])

    df["wap1"] = calc_wap(bpr1, bsz1, apr1, asz1)
    df["wap2"] = calc_wap(bpr2, bsz2, apr2, asz2)

    df["log_return_wap1"] = df.groupby("time_id")["wap1"].apply(log_return)
    df["log_return_wap2"] = df.groupby("time_id")["wap2"].apply(log_return)

    df["wap_balance"] = df["wap1"] - df["wap2"]

    df["price_spread1"] = (apr1 - bpr1) / ((apr1 + bpr1) / 2)
    df["price_spread2"] = (apr2 - bpr2) / ((apr2 + bpr2) / 2)
    df["bid_spread"] = bpr1 - bpr2
    df["ask_spread"] = apr2 - apr1
    df["bid_ask_spread"] = (bpr1 - bpr2) - (apr1 - apr2)
    df["total_volume"] = (asz1 + asz2) + (bsz1 + bsz2)
    df["volume_imbalance"] = (asz1 + asz2) - (bsz1 + bsz2)

    agg = {
        "time_id": ["count"],
        "wap1": ["mean", "std", max_sub_min],
        "wap2": ["mean", "std", max_sub_min],
        "log_return_wap1": ["mean", "std", abs_sum, rv],
        "log_return_wap2": ["mean", "std", abs_sum, rv],
        "wap_balance": ["mean", "std", abs_sum],
        "price_spread1": ["sum", "mean", "std"],
        "price_spread2": ["sum", "mean", "std"],
        "bid_spread": ["sum", "mean", "std"],
        "ask_spread": ["sum", "mean", "std"],
        "bid_ask_spread": ["mean", "std", abs_sum],
        "total_volume": ["sum", "mean", "std"],
        "volume_imbalance": ["mean", "std", abs_sum],
    }

    group_df = df.groupby(["time_id"]).agg(agg)
    group_df.columns = [f"book_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)

    # realized volatility diff
    df["5min"] = pd.cut(df["seconds_in_bucket"], bins=np.linspace(0, 600, 3), right=False).cat.codes
    temp = df.groupby(["time_id", "5min"]).agg({
        "log_return_wap1": [rv],
        "log_return_wap2": [rv],
    })
    temp.columns = [f"book_{f[0]}_{f[1]}" for f in temp.columns]
    temp.reset_index(inplace=True)
    for feat in ["book_log_return_wap1_rv", "book_log_return_wap2_rv"]:
        temp[f"{feat}_shift"] = temp[feat].shift(1)

    temp = temp.loc[temp["5min"] == 1]
    temp.reset_index(drop=True, inplace=True)

    for feat in ["book_log_return_wap1_rv", "book_log_return_wap2_rv"]:
        temp[f"{feat}_diff"] = temp[feat] - temp[f"{feat}_shift"]
        temp[f"{feat}_pct_change"] = (temp[feat] - temp[f"{feat}_shift"]) / (temp[f"{feat}_shift"] + 1e-6)

        temp.drop([feat, f"{feat}_shift"], axis=1, inplace=True)
    del temp["5min"]

    group_df = group_df.merge(temp, on="time_id", how="left")

    last_seconds = [300]
    for second in last_seconds:
        second = 600 - second

        group_df_sec = df.query(f"seconds_in_bucket >= {second}").groupby(["time_id"]).agg(agg)
        group_df_sec.columns = [f"book_last_{second}_{f[0]}_{f[1]}" for f in group_df_sec.columns]
        group_df_sec.reset_index(inplace=True)

        group_df = group_df.merge(group_df_sec, on="time_id", how="left")

    stock_id = file_path.split("=")[1]
    group_df["row_id"] = group_df["time_id"].map(lambda x: f"{stock_id}-{x}")
    del group_df["time_id"]

    return group_df


def get_trade_feat(file_path):
    df = pd.read_parquet(file_path)

    df["amount"] = df["price"] * df["size"]
    df["log_return"] = df.groupby("time_id")["price"].apply(log_return)
    df["size_div_order_count"] = df["price"] / df["order_count"]
    df["seconds_diff"] = df.groupby("time_id")["seconds_in_bucket"].diff()

    agg = {
        "time_id": ["count"],
        "price": ["mean", "std", max_sub_min],
        "log_return": ["mean", "std", abs_sum, rv],
        "size": ["sum", max_sub_min],
        "order_count": ["sum", max_sub_min],
        "amount": ["sum", "mean", "std", max_sub_min],
        "size_div_order_count": ["mean", "std"],
        "seconds_diff": ["mean", "std", abs_sum, rv],
    }

    group_df = df.groupby("time_id").agg(agg)
    group_df.columns = [f"trade_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)
    group_df.fillna(0, inplace=True)

    # realized volatility diff
    df["5min"] = pd.cut(df["seconds_in_bucket"], bins=np.linspace(0, 600, 3), right=False).cat.codes
    temp = df.groupby(["time_id", "5min"]).agg({
        "log_return": [rv],
    })
    temp.columns = [f"trade_{f[0]}_{f[1]}" for f in temp.columns]
    temp.reset_index(inplace=True)
    for feat in ["trade_log_return_rv"]:
        temp[f"{feat}_shift"] = temp[feat].shift(1)

    temp = temp.loc[temp["5min"] == 1]
    temp.reset_index(drop=True, inplace=True)

    for feat in ["trade_log_return_rv"]:
        temp[f"{feat}_diff"] = temp[feat] - temp[f"{feat}_shift"]
        temp[f"{feat}_pct_change"] = (temp[feat] - temp[f"{feat}_shift"]) / (temp[f"{feat}_shift"] + 1e-6)

        temp.drop([feat, f"{feat}_shift"], axis=1, inplace=True)
    del temp["5min"]

    group_df = group_df.merge(temp, on="time_id", how="left")

    last_seconds = [300]
    for second in last_seconds:
        second = 600 - second

        group_df_sec = df.query(f"seconds_in_bucket >= {second}").groupby(["time_id"]).agg(agg)
        group_df_sec.columns = [f"trade_last_{second}_{f[0]}_{f[1]}" for f in group_df_sec.columns]
        group_df_sec.reset_index(inplace=True)
        group_df_sec.fillna(0, inplace=True)

        group_df = group_df.merge(group_df_sec, on="time_id", how="left")

    stock_id = file_path.split("=")[1]
    group_df["row_id"] = group_df["time_id"].map(lambda x: f"{stock_id}-{x}")
    del group_df["time_id"]

    return group_df


def get_time_stock_feat(df):
    # Get realized volatility columns
    vol_cols = [
        "book_log_return_wap1_rv",
        "book_log_return_wap2_rv",
        "trade_log_return_rv",
        "trade_seconds_diff_rv",
        "book_last_300_log_return_wap1_rv",
        "book_last_300_log_return_wap2_rv",
        "trade_last_300_log_return_rv",
        "trade_last_300_seconds_diff_rv",
    ]

    # group by the stock id
    df_stock_id = df.groupby(["stock_id"])[vol_cols].agg(["mean", "std", "skew"])
    df_stock_id.columns = [f"stock_id_{f[0]}_{f[1]}" for f in df_stock_id.columns]
    df_stock_id.reset_index(inplace=True)

    # group by the time id
    df_time_id = df.groupby(["time_id"])[vol_cols].agg(["mean", "std", "skew"])
    df_time_id.columns = [f"time_id_{f[0]}_{f[1]}" for f in df_time_id.columns]
    df_time_id.reset_index(inplace=True)

    # merge with original dataframe
    df = df.merge(df_time_id, on="time_id", how="left")
    df = df.merge(df_stock_id, on="stock_id", how="left")

    return df


def get_cluster_feat(df, mapping):
    df["cluster"] = df["stock_id"].map(mapping)

    usecols = [
        "book_log_return_wap1_realized_volatility",
        "book_total_volume_mean",
        "trade_size_sum",
        "trade_order_count_sum",
        "book_price_spread1_mean",
        "book_bid_spread_mean",
        "book_ask_spread_mean",
        "book_volume_imbalance_mean",
        "book_bid_ask_spread_mean",
    ]
    group_df = df.groupby(["time_id", "cluster"], as_index=False)[usecols].mean()
    group_df = group_df.pivot(index="time_id", columns="cluster")
    group_df.columns = [f"{f[0]}_c{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)

    df = df.merge(group_df, on="time_id", how="left")

    return df


def get_feat(stock_ids, is_train=True, n_jobs=-1):
    data_dir = "../input/optiver-realized-volatility-prediction"

    def ufunc(stock_id):
        if is_train:
            file_path_book = f"{data_dir}/book_train.parquet/stock_id={stock_id}"
            file_path_trade = f"{data_dir}/trade_train.parquet/stock_id={stock_id}"
        else:
            file_path_book = f"{data_dir}/book_test.parquet/stock_id={stock_id}"
            file_path_trade = f"{data_dir}/trade_test.parquet/stock_id={stock_id}"

        return pd.merge(get_book_feat(file_path_book), get_trade_feat(file_path_trade), on="row_id", how="left")

    df = Parallel(n_jobs=n_jobs, verbose=1)(delayed(ufunc)(stock_id) for stock_id in stock_ids)
    df = pd.concat(df, ignore_index=True)

    return df
