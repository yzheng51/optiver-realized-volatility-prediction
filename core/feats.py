import numpy as np
import pandas as pd


def calc_wap(bid_price, bid_size, ask_price, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def get_book_feat(file_path):
    df = pd.read_parquet(file_path)

    bpr1, bsz1, apr1, asz1 = (df[col].values for col in ["bid_price1", "bid_size1", "ask_price1", "ask_size1"])
    bpr2, bsz2, apr2, asz2 = (df[col].values for col in ["bid_price2", "bid_size2", "ask_price2", "ask_size2"])

    # calculate return etc
    df["wap1"] = calc_wap(bpr1, bsz1, apr1, asz1)
    df["wap2"] = calc_wap(bpr2, bsz2, apr2, asz2)

    df["log_return1"] = df.groupby("time_id")["wap1"].apply(log_return)
    df["log_return2"] = df.groupby("time_id")["wap2"].apply(log_return)

    df["wap_balance"] = np.abs(df["wap1"].values - df["wap2"].values)

    df["price_spread1"] = (apr1 - bpr1) / ((apr1 + bpr1) / 2)
    df["price_spread2"] = (apr2 - bpr2) / ((apr2 + bpr2) / 2)
    df["bid_spread"] = bpr1 - bpr2
    df["ask_spread"] = apr1 - apr2
    df["bid_ask_spread"] = np.abs((bpr1 - bpr2) - (apr1 - apr2))
    df["total_volume"] = (asz1 + asz2) + (bsz1 + bsz2)
    df["volume_imbalance"] = np.abs((asz1 + asz2) - (bsz1 + bsz2))

    # dict for aggregate
    agg = {
        "wap1": ["sum", "mean", "std"],
        "wap2": ["sum", "mean", "std"],
        "log_return1": ["sum", "mean", "std", realized_volatility],
        "log_return2": ["sum", "mean", "std", realized_volatility],
        "wap_balance": ["sum", "mean", "std"],
        "price_spread1": ["sum", "mean", "std"],
        "price_spread2": ["sum", "mean", "std"],
        "bid_spread": ["sum", "mean", "std"],
        "ask_spread": ["sum", "mean", "std"],
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

    agg = {
        "log_return": ["sum", "mean", "std", realized_volatility],
        "seconds_in_bucket": ["nunique"],
        "size": ["sum", "mean", "std"],
        "order_count": ["sum", "mean", "std"],
    }

    group_df = df.groupby("time_id").agg(agg)
    group_df.columns = [f"trade_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)

    # groupby / last XX seconds
    last_seconds = [300]

    for second in last_seconds:
        second = 600 - second

        group_df_sec = df.query(f"seconds_in_bucket >= {second}").groupby(["time_id"]).agg(agg)
        group_df_sec.columns = [f"trade_last_{second}_{f[0]}_{f[1]}" for f in group_df_sec.columns]
        group_df_sec.reset_index(inplace=True)

        group_df = group_df.merge(group_df_sec, on="time_id", how="left")

    stock_id = file_path.split("=")[1]
    group_df["row_id"] = group_df["time_id"].map(lambda x: f"{stock_id}-{x}")
    del group_df["time_id"]

    return group_df


def get_time_id_feat(df):
    # Get realized volatility columns
    vol_cols = [
        "book_log_return1_realized_volatility", "book_log_return2_realized_volatility",
        "book_last_300_log_return1_realized_volatility", "book_last_300_log_return2_realized_volatility",
        "trade_log_return_realized_volatility", "trade_last_300_log_return_realized_volatility"
    ]

    # group by the time id
    group_df = df.groupby(["time_id"])[vol_cols].agg(["mean", "std", "max", "min", "skew"])
    group_df.columns = [f"time_id_{f[0]}_{f[1]}" for f in group_df.columns]
    group_df.reset_index(inplace=True)

    # merge with original dataframe
    df = df.merge(group_df, on="time_id", how="left")

    return df
