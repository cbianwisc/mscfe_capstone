import datetime
import os

import numpy as np
import pandas as pd

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import split_on_daily_basis
from src_py.data_retriever_and_processor.data_preprocessor_time_series import mark_time_from_datetime
from src_py.data_retriever_and_processor.data_reader import read_data, calculate_log_return
from src_py.data_retriever_and_processor.input_retriever import standardize_data


def read_vix_data(years=None):
    curr_dir = os.getcwd()
    home_dir = curr_dir[:curr_dir.index('mscfe_capstone') + len('mscfe_capstone')]
    datasets_dir = home_dir + '\\Datasets'
    if (years is not None) and (years == 5):
        five_yr_data_dir = datasets_dir + '\\5YearsOfData'
        five_yr_20min_data_dir = five_yr_data_dir + '\\VIX_CBOE_USD_5Y_20m_ending_05052023.csv'
        df_raw_data = read_data(five_yr_20min_data_dir)
    elif (years is not None) and (years == 4):
        four_yr_data_dir = datasets_dir + '\\4YearsOfData'
        four_yr_10min_data_dir = four_yr_data_dir + '\\VIX_CBOE_USD_4Y_10m_ending_05052023.csv'
        df_raw_data = read_data(four_yr_10min_data_dir)
    else:
        df_raw_data = pd.DataFrame()
        print('No VIX data available for the selected time span')
    return df_raw_data


def read_time_stamp(df: pd.DataFrame) -> pd.DataFrame:
    series_orig_time_stamp = df['Time']
    series_time_stamp = series_orig_time_stamp.map(lambda x: x.rstrip(' US/Central'))
    series_datetime_central = series_time_stamp.map(lambda x: datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S'))
    series_datetime = series_datetime_central + datetime.timedelta(hours=1)
    df['datetime'] = series_datetime
    df = df.drop(columns=['Time'])
    return df


def filter_only_early_morning_returns(df: pd.DataFrame) -> pd.DataFrame:
    df_early_morning_returns = df.loc[df['datetime'].dt.hour == 3]
    return df_early_morning_returns.replace([np.inf, -np.inf], 0.0)


def reformat_daily_inputs_vix(df):
    df_daily_marked = split_on_daily_basis(df)
    df_time_marked = mark_time_from_datetime(df_daily_marked)
    df_time_marked = df_time_marked.replace([np.inf, -np.inf], 0.0)
    df_pivoted = df_time_marked.pivot(index='date', columns='time', values=['Open_return', 'Close_return'])
    df_pivoted = df_pivoted.fillna(0.0)
    df_pivoted.columns = ['VIX_' + x[0] + '_' + str(x[1]) for x in df_pivoted.columns.tolist()]
    return df_pivoted


def combine_columns(df):
    df_ret = df.copy()
    df_ret['VIX_Open_return_03:00:00'] = df_ret['VIX_Open_return_03:00:00'] + df_ret['VIX_Open_return_03:15:00']
    df_ret['VIX_Close_return_03:00:00'] = df_ret['VIX_Close_return_03:00:00'] + df_ret['VIX_Close_return_03:15:00']
    df_ret = df_ret.drop(columns=['VIX_Open_return_03:15:00', 'VIX_Close_return_03:15:00'])
    return  df_ret

def prepare_daily_inputs_vix(years):
    df = read_vix_data(years=years)
    df = read_time_stamp(df)
    df = calculate_log_return(df, ['Open', 'Close'])
    df = filter_only_early_morning_returns(df)
    df = reformat_daily_inputs_vix((df))
    df = combine_columns(df)
    df = standardize_data(df)
    return df


if __name__ == "__main__":
    df = prepare_daily_inputs_vix(5)
    print(df)
