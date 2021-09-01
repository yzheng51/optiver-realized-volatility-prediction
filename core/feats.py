import numpy as np
import pandas as pd


def calc_wap(bid_price, bid_size, ask_price, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def abs_sum(series):
    return np.sum(np.abs(series))


def max_sub_min(series):
    return np.max(series) - np.min(series)


def get_book_feat(file_path):
    df = pd.read_parquet(file_path)

    bpr1, bsz1, apr1, asz1 = (df[col].values for col in ["bid_price1", "bid_size1", "ask_price1", "ask_size1"])
    bpr2, bsz2, apr2, asz2 = (df[col].values for col in ["bid_price2", "bid_size2", "ask_price2", "ask_size2"])

    df["wap1"] = calc_wap(bpr1, bsz1, apr1, asz1)
    df["wap2"] = calc_wap(bpr2, bsz2, apr2, asz2)

    df["log_return_wap1"] = df.groupby("time_id")["wap1"].apply(log_return)
    df["log_return_wap2"] = df.groupby("time_id")["wap2"].apply(log_return)

    df["wap_balance"] = abs(df["wap1"] - df["wap2"])

    df["price_spread1"] = (apr1 - bpr1) / ((apr1 + bpr1) / 2)
    df["price_spread2"] = (apr2 - bpr2) / ((apr2 + bpr2) / 2)
    df["bid_spread"] = bpr1 - bpr2
    df["ask_spread"] = apr1 - apr2
    df["bid_ask_spread"] = abs((bpr1 - bpr2) - (apr1 - apr2))
    df["total_volume"] = (asz1 + asz2) + (bsz1 + bsz2)
    df["volume_imbalance"] = abs((asz1 + asz2) - (bsz1 + bsz2))

    # dict for aggregate
    agg = {
        "time_id": ["count"],
        "wap1": ["mean", max_sub_min, "std"],
        "wap2": ["mean", max_sub_min, "std"],
        "log_return_wap1": [abs_sum, "mean", "std", realized_volatility],
        "log_return_wap2": [abs_sum, "mean", "std", realized_volatility],
        "wap_balance": ["sum", "mean", "std"],
        "price_spread1": ["sum", "mean", "std"],
        "price_spread2": ["sum", "mean", "std"],
        "bid_spread": ["sum", "mean", "std"],
        "ask_spread": ["sum", "mean", "std"],
        "bid_ask_spread": ["sum", "mean", "std"],
        "total_volume": ["sum", "mean", "std"],
        "volume_imbalance": ["sum", "mean", "std"],
    }

    # groupby / all seconds
    group_df = df.groupby(["time_id"]).agg(agg)
    group_df.columns = [f"book_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)

    # groupby / last XX seconds
    last_seconds = [300]

    for second in last_seconds:
        second = 600 - second

        group_df_sec = df.query(f"seconds_in_bucket >= {second}").groupby(["time_id"]).agg(agg)
        group_df_sec.columns = [f"book_last_{second}_{f[0]}_{f[1]}" for f in group_df_sec.columns]
        group_df_sec.reset_index(inplace=True)

        group_df = group_df.merge(group_df_sec, on="time_id", how="left")

    # create row_id
    stock_id = file_path.split("=")[1]
    group_df["row_id"] = group_df["time_id"].map(lambda x: f"{stock_id}-{x}")
    del group_df["time_id"]

    return group_df


def get_trade_feat(file_path):
    df = pd.read_parquet(file_path)
    df["log_return"] = df.groupby("time_id")["price"].apply(log_return)
    df["size_div_order_count"] = df["price"] / df["order_count"]
    df["seconds_in_bucket_diff"] = df["seconds_in_bucket"].diff()
    df["total_price"] = df["price"] * df["size"]

    agg = {
        "time_id": ["count"],
        "price": [max_sub_min, "mean", "std"],
        "total_price": ["sum", max_sub_min, "mean", "std"],
        "log_return": [abs_sum, "mean", "std", realized_volatility],
        "seconds_in_bucket_diff": [abs_sum, "mean", "std", realized_volatility],
        "size": ["sum", max_sub_min],
        "size_div_order_count": ["mean", "std"],
        "order_count": ["sum", max_sub_min],
    }

    group_df = df.groupby("time_id").agg(agg)
    group_df.columns = [f"trade_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)
    group_df.fillna(0, inplace=True)

    # groupby / last XX seconds
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
        "book_log_return_wap1_realized_volatility",
        "book_log_return_wap2_realized_volatility",
        "trade_log_return_realized_volatility",
        "trade_seconds_in_bucket_diff_realized_volatility",
        "book_last_300_log_return_wap1_realized_volatility",
        "book_last_300_log_return_wap2_realized_volatility",
        "trade_last_300_log_return_realized_volatility",
        "trade_last_300_seconds_in_bucket_diff_realized_volatility",
    ]

    # group by the stock id
    df_stock_id = df.groupby(["stock_id"])[vol_cols].agg(["mean", "std"])
    df_stock_id.columns = [f"stock_id_{f[0]}_{f[1]}" for f in df_stock_id.columns]
    df_stock_id.reset_index(inplace=True)

    # group by the time id
    df_time_id = df.groupby(["time_id"])[vol_cols].agg(["mean", "std"])
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
