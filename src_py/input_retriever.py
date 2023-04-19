import math

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing

from src_py.data_reader import START_DATE, END_DATE


def retreive_yfiance_data(tickers: []):
    df_yf = yf.download(tickers, start=START_DATE, end=END_DATE)
    return df_yf


def calculate_return_or_change(df: pd.DataFrame):
    df_ret = pd.DataFrame()
    for col in df.columns:
        df_col = df[col[0]]
        for ticker in df_col.columns:
            series_curr = df_col[ticker]
            series_last = series_curr.shift(1)
            series_change = series_curr.div(series_last, fill_value=0.0)
            df_ret[ticker + '_' + col[0] + '_change'] = series_change - 1
    df_ret = df_ret.iloc[1:, :].copy()
    df_ret = df_ret.fillna(0.0)
    df_ret = df_ret.replace([np.inf, -np.inf], 0.0)
    return df_ret


def remove_empty_column(df: pd.DataFrame):
    df_ret = pd.DataFrame()
    for col in df.columns:
        if ~(df[col] == 0).all():
            df_ret[col] = df[col]
    return df_ret


def standardize_data(df: pd.DataFrame):
    scaler = preprocessing.StandardScaler().fit(df)
    np_array_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(
        data=np_array_scaled,
        index=df.index,
        columns=df.columns
    )
    return df_scaled


if __name__ == "__main__":
    df_yf = retreive_yfiance_data(['^VIX', '^VVIX', 'SPY', 'TQQQ', '^IXIC', 'NQ=F'])
    df_return = calculate_return_or_change(df_yf)
    df_not_empty = remove_empty_column(df_return)
    df_standardized = standardize_data(df_not_empty)
    print(df_standardized)
