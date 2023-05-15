import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing


def retreive_yfiance_data(tickers: [], start_date: datetime.date, end_date: datetime.date):
    df_yf = yf.download(tickers, start=start_date, end=end_date)
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
        if ~(df[col] == 0.0).all():
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


def convert_index_from_datetime_to_date(df: pd.DataFrame):
    index_as_datetime = df.index
    df.index = index_as_datetime.date
    return df


def retrieve_input_data(start_date: datetime.date, end_date: datetime.date):
    df_yf = retreive_yfiance_data(['^VIX', '^VVIX', 'SPY', 'TQQQ', '^IXIC', 'NQ=F'],
                                  start_date=start_date,
                                  end_date=end_date)
    df_return = calculate_return_or_change(df_yf)
    df_not_empty = remove_empty_column(df_return)
    df_standardized = standardize_data(df_not_empty)
    df_index_converted = convert_index_from_datetime_to_date(df_standardized)
    return df_index_converted


if __name__ == "__main__":
    df_retrieved = retrieve_input_data(datetime.date(2018, 5, 7), datetime.date(2023, 5, 5))
    print(df_retrieved)
